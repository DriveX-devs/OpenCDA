# -*- coding: utf-8 -*-
"""
Perception module base.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import weakref
import sys
import time
import math
import carla
import cv2
import numpy as np
import open3d as o3d
import os
import csv
import random
import threading
import pandas as pd

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.obstacle_pedestrian import \
    ObstacleVRU
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion , o3d_camera_lidar_fusion2, o3d_camera_lidar_fusion3
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean
# from opencda.customize.v2x.LDMutils import compute_IoU

class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.image = None
        self.timstamp = None
        self.frame = 0
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        # camera attributes
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35

        carla_location = carla.Location(x=carla_location.x + x,
                                        y=carla_location.y + y,
                                        z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)
        image = image.reshape((self.image_height, self.image_width, 4))
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image
        self.frame = event.frame
        self.timestamp = event.timestamp


class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))
        blueprint.set_attribute(
            'dropoff_general_rate', str(
                config_yaml['dropoff_general_rate']))
        blueprint.set_attribute(
            'dropoff_intensity_limit', str(
                config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute(
            'dropoff_zero_intensity', str(
                config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute(
            'noise_stddev', str(
                config_yaml['noise_stddev']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.data = None
        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        # Create a directory to store the output files
        # self.output_path = "lidar_output"
        # os.makedirs(self.output_path, exist_ok=True)
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # 2. Update the Open3D point cloud object
        # We only need the (x, y, z) coordinates for the geometry
        # points = data[:, :3]
        # self.o3d_pointcloud.points = o3d.utility.Vector3dVector(points)
        #
        # # 3. Save the point cloud to a .pcd file
        # # Use the frame number to create a unique filename for each frame
        # filename = os.path.join(self.output_path, f"{event.frame:06d}.pcd")
        # o3d.io.write_point_cloud(filename, self.o3d_pointcloud)

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # set attribute based on the configuration
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # lidar data
        self.points = None
        self.obj_idx = None
        self.obj_tag = None

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # shape:(n, 6)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # (x, y, z, intensity)
        self.points = np.array([data['x'], data['y'], data['z']]).T
        self.obj_tag = np.array(data['ObjTag'])
        self.obj_idx = np.array(data['ObjIdx'])

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class RadarSensor:
    """
    Radar manager for vehicle or infrastructure.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : list
        Indicates the sensor position relative to vehicle or infrastructure,
        [x, y, z, yaw].

    Attributes
    ----------
    detections : list
        Current list of detected objects.
    sensor : carla.Sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, global_position=None):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = world.get_blueprint_library().find('sensor.other.radar')
        blueprint.set_attribute('horizontal_fov', '30')
        blueprint.set_attribute('vertical_fov', '30')
        blueprint.set_attribute('range', '100')

        # spawn sensor
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        self.detections = []
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: RadarSensor._on_radar_event(weak_self, event))

    @staticmethod
    def spawn_point_estimation(relative_position, global_position=None):
        pitch = 0
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2]
            )
            pitch = -35

        carla_location = carla.Location(
            x=carla_location.x + x,
            y=carla_location.y + y,
            z=carla_location.z + z
        )

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    @staticmethod
    def _on_radar_event(weak_self, event):
        """Callback method for when radar data is received from the sensor."""
        self = weak_self()
        if not self:
            return

        self.detections = []
        for detection in event:
            detection_data = {
                'velocity': detection.velocity,
                'azimuth': detection.azimuth,
                'altitude': detection.altitude,
                'depth': detection.depth
            }
            self.detections.append(detection_data)


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):
        self.vehicle = vehicle
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self._map = self.carla_world.get_map()
        self.id = infra_id if infra_id is not None else vehicle.id

        self.activate = config_yaml['activate']
        self.camera_visualize = config_yaml['camera']['visualize']
        self.camera_num = config_yaml['camera']['num']
        self.lidar_visualize = config_yaml['lidar']['visualize']
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None

        self.cav_world = weakref.ref(cav_world)()
        ml_manager = cav_world.ml_manager

        if self.activate and data_dump:
            sys.exit("When you dump data, please deactivate the "
                     "detection function for precise label.")

        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in'
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        if self.activate or self.camera_visualize:
            self.rgb_camera = []
            mount_position = config_yaml['camera']['positions']
            assert len(mount_position) == self.camera_num, \
                "The camera number has to be the same as the length of the" \
                "relative positions list"

            for i in range(self.camera_num):
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position))

        else:
            self.rgb_camera = None

        self.lidar = LidarSensor(vehicle,
                                 self.carla_world,
                                 config_yaml['lidar'],
                                 self.global_position)
        if self.lidar_visualize:
            self.o3d_vis = o3d_visualizer_init(self.id)
        else:
            self.o3d_vis = None

        self.radar = RadarSensor(vehicle, self.carla_world, self.global_position)

        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if data_dump:
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}
        # traffic light detection related
        self.traffic_thresh = config_yaml['traffic_light_thresh'] \
            if 'traffic_light_thresh' in config_yaml else 50

        # --- CSV Configuration ---
        self.EXCLUDED_PED_COUNT_CSV_FILE = "excluded_pedestrian_counts.csv"
        self.EXCLUDED_PED_COUNT_HEADERS = [
            "timestamp",
            "perceiver_id",
            "excluded_pedestrian_count"
        ]
        self.confidence_csv_file = "Confidence_level_all_PO.csv"
        self.confidence_csv_header = [
            "timestamp",
            "confidence",
            "ID"
        ]
        # self.detection_csv_file = "Detection_parameters.csv"
        # self.detection_csv_header = [
        #     "true_positives",
        #     "false_negatives",
        #     "false_positives"
        # ]

        # --- Threading Lock for CSV writing ---
        # If you have other CSV writing operations, ensure locks are used appropriately
        # to prevent deadlocks or if a single lock can manage all CSV access.
        # For this specific file, we use this lock:
        self.excluded_ped_count_csv_lock = threading.Lock()
        self.confidence_csv_lock = threading.Lock()

    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.

        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """
        return a.get_location().distance(self.ego_pos.location)

    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.
        Implementation for pedestrians detection ongoing.

        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos

        objects = {'vehicles': [],
                   'traffic_lights': [],
                   'VRU': []}

        if not self.activate:
            objects = self.deactivate_mode(objects)

        else:
            objects = self.activate_mode(objects)

        self.count += 1

        return objects

    def radar_detect(self, objects):

        velocity_range = 7.5  # m/s
        current_rot = self.radar.sensor.get_transform().rotation

        # Step 1: Extract 3D Points
        points = []
        for detect in self.radar.detections:
            azi = math.degrees(detect['azimuth'])
            alt = math.degrees(detect['altitude'])
            fw_vec = carla.Vector3D(x=detect['depth'] - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)
            # point = self.radar.sensor.get_transform().location + fw_vec
            point = fw_vec
            points.append([point.x, point.y, point.z, detect['velocity']])

        if not points:
            return  # No points to process

        # Convert to numpy array
        points_np = np.array(points)

        # Step 2: Cluster the Points
        clustering = DBSCAN(eps=2.0, min_samples=7).fit(points_np)
        labels = clustering.labels_

        # Step 3: Generate Colors for Each Cluster
        unique_labels = set(labels)
        colors = {}
        for label in unique_labels:
            if label == -1:  # Noise points
                colors[label] = (255, 255, 255)  # White
            if label == 0:
                colors[label] = (255, 0, 0)
            if label == 1:
                colors[label] = (0, 255, 0)
            if label == 2:
                colors[label] = (0, 0, 255)

        # Step 4: Draw the Points with Cluster Colors
        for point, label in zip(points, labels):
            #point = point + self.radar.sensor.get_transform().location
            point[0] = point[0] + self.radar.sensor.get_transform().location.x
            point[1] = point[1] + self.radar.sensor.get_transform().location.y
            point[2] = point[2] + self.radar.sensor.get_transform().location.z
            if point[2] < 0.5 or point[2] > 2.5 or point[3] < (5 - self.vehicle.get_velocity().x) or label == -1:
                continue
            # This messes up the yolo detection
            #r, g, b = colors[label]
            # self.carla_world.debug.draw_point(
            #     carla.Location(x=point[0], y=point[1], z=point[2]),
            #     size=0.075,
            #     life_time=0.1,
            #     persistent_lines=False,
            #     color=carla.Color(r, g, b))

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        ped_list = world.get_actors()
        ped_list = [p for p in ped_list if self.dist(p) < 50 and p.id != self.id]

        # Step 5: Create and Draw Bounding Boxes
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            cluster_points = points_np[labels == label]
            # skip points with point[0] < 0.5 or point[0] > 2.5 or point[3] < (5 - self.vehicle.get_velocity().x)
            cluster_points = cluster_points[cluster_points[:, 3] > (5 - self.vehicle.get_velocity().x)]

            if len(cluster_points) == 0:
                continue

            min_coords = cluster_points.min(axis=0)
            max_coords = cluster_points.max(axis=0)

            # Create Open3D AxisAlignedBoundingBox
            o3d_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_coords[:3], max_bound=max_coords[:3])

            # Create the 8 corners of the bounding box
            bbox_corners = [
                [min_coords[0], min_coords[1], min_coords[2]],
                [min_coords[0], min_coords[1], max_coords[2]],
                [min_coords[0], max_coords[1], min_coords[2]],
                [min_coords[0], max_coords[1], max_coords[2]],
                [max_coords[0], min_coords[1], min_coords[2]],
                [max_coords[0], min_coords[1], max_coords[2]],
                [max_coords[0], max_coords[1], min_coords[2]],
                [max_coords[0], max_coords[1], max_coords[2]],
            ]

            bbox_corners = np.array(bbox_corners)
            bbox_corners[:, 0] += self.radar.sensor.get_transform().location.x
            bbox_corners[:, 1] += self.radar.sensor.get_transform().location.y
            bbox_corners[:, 2] += self.radar.sensor.get_transform().location.z



            obstacle_vehicle = ObstacleVehicle(bbox_corners, o3d_bbx, confidence=0.71)
            obstacle_vehicle.set_velocity(
                 carla.Vector3D(self.vehicle.get_velocity().x + cluster_points.mean(axis=0)[3], 0, 0))

            # obstacle_VRU = ObstacleVRU(bbox_corners, o3d_bbx, confidence=0.71)
            # obstacle_VRU.set_velocity(
            #     carla.Vector3D(self.vehicle.get_velocity().x + cluster_points.mean(axis=0)[3], 0, 0))

            for v in vehicle_list:
                loc = v.get_location()
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                    abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.carla_id = v.id

            # for p in in ped_list:
            #     p_loc = p.get_location()

            objects['vehicles'].append(obstacle_vehicle)


        return objects


    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # retrieve current cameras and lidar data
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))


        # yolo detection
        init = time.time_ns()
        yolo_detection = self.ml_manager.object_detector(rgb_images)
        yolo_time = time.time_ns()
        #print('yolo detection time [ms]: ' + str((yolo_time - init) / 1e6))
        # rgb_images for drawing
        rgb_draw_images = []

        data_copy = np.copy(self.lidar.data)

        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, data_copy, np.array(
                    rgb_camera.image))

            rgb_image, projected_radar = st.project_radar_to_camera(
                self.radar.sensor, rgb_camera.sensor, self.radar.detections, np.array(rgb_image))

            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion3(
                objects,
                yolo_detection.xyxy[i],
                data_copy,
                projected_lidar,
                self.lidar.sensor,
                self.ego_pos)

            # objects = o3d_camera_lidar_fusion2(
            #     objects,
            #     yolo_detection.xyxy[i],
            #     data_copy,
            #     projected_lidar,
            #     self.lidar.sensor,
            #     self.ego_pos)

            # calculate the speed. current we retrieve from the server
            # directly.
            # this function is used also to set the id
            self.speed_retrieve(objects)

        #self.radar_detect(objects)


        fusion_time = time.time_ns()
        #print('fusion time [ms]: ' + str((fusion_time - yolo_time) / 1e6))
        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.8, fy=0.8)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)
            cv2.waitKey(1)

        for key in objects:
            if key == 'static':
                continue
            if key == "vehicle":
                objects[key] = [item for item in objects[key] if item.confidence >= 0.7]
            if key == "VRU":
                current_list = objects[key]
                vrus_to_be_excluded = [
                    item for item in current_list if item.confidence < 0.4 and item.carla_id != -1]
                excluded_count = len(vrus_to_be_excluded)
                self.log_excluded_pedestrian_count(self.id, excluded_count)
                for item in current_list:
                    self.log_confidence_level(item.confidence, item.carla_id)

                objects[key] = [item for item in objects[key] if item.confidence >= 0.4]

            duplicate_indices = set()
            # Iterate through the objects to check for duplicates
            for i in range(len(objects[key])):
                for j in range(i + 1, len(objects[key])):
                    dist = math.sqrt(pow(objects[key][i].location.x - objects[key][j].location.x, 2)
                                     + pow(objects[key][i].location.y - objects[key][j].location.y, 2))
                    if (dist < 2 and key == 'vehicles') or objects[key][i].carla_id == objects[key][j].carla_id:
                        # if (objects['vehicles'][i].bounding_box.extent.x*objects['vehicles'][i].bounding_box.extent.y) > \
                        #         (objects['vehicles'][j].bounding_box.extent.x * objects['vehicles'][j].bounding_box.extent.y):
                        if objects[key][i].confidence > objects[key][j].confidence:
                            duplicate_indices.add(j)
                        else:
                            duplicate_indices.add(i)

            # Remove duplicate objects from the list
            for index in sorted(duplicate_indices, reverse=True):
                objects[key].pop(index)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(data_copy, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        #print('Matching time [ms]: ' + str((time.time_ns() - fusion_time) / 1e6))
        return objects

    def deactivate_mode(self, objects):
        """
        Object detection using server information directly.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        # todo: hard coded
        thresh = 30 if not self.data_dump else 30

        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                        v.id != self.id]

        # use semantic lidar to filter out vehicles out of the range
        if self.data_dump:
            vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects.update({'vehicles': vehicle_list})

        if self.camera_visualize:
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def getGTobjects(self):
        """
        Object detection using server information directly.

        Returns
        -------
         objects: dict
            Object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        VRU_list = world.get_actors().filter("*pedestrian*")
        # todo: hard coded
        thresh = 75

        if self.ego_pos:
            vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                            v.id != self.id]
            VRU_list = [p for p in VRU_list if self.dist(p) < thresh and
                            p.id != self.id]
        else:
            vehicle_list = [v for v in vehicle_list if v.id != self.id]
            VRU_list = [p for p in VRU_list if p.id != self.id]

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    None) for v in vehicle_list]
            VRU_list = [
                ObstacleVRU(
                    None,
                    None,
                    p,
                    self.lidar.sensor,
                    None) for p in VRU_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
            VRU_list = [
                ObstacleVRU(
                    None,
                    None,
                    p,
                    None,
                    self.cav_world.sumo2carla_ids) for p in VRU_list]

        objects = {'vehicles': vehicle_list, 'VRU': VRU_list}
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)

        return objects

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.

        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        # label 10 is the vehicle
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image

    def speed_retrieve(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return
        if 'VRU' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:

            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())

                    # the case where the obstacle vehicle is controlled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)

                    obstacle_vehicle.set_carla_id(v.id)

        allactor = world.get_actors()
        VRU_list = []
        blueprint_library = world.get_blueprint_library()
        for actor in allactor:
            try:
                bp = blueprint_library.find(actor.type_id)
            except:
                continue
            # actor = world.get_actor(actor.id)
            # actor.attributes.object_type
            # actor type id has no motorcycle or bicycle
            # todo: check for blueprint attributes
            if bp.has_attribute("number_of_wheels"):
                if actor.attributes['number_of_wheels'] == '2':
                    VRU_list.append(actor)
            elif 'pedestrian' in actor.type_id:
                VRU_list.append(actor)
        VRU_list = [p for p in VRU_list if self.dist(p) < 40 and
                        p.id != self.id]

        # -----Hungarian algorithm-----
        if objects['VRU']:
            MAX_DISTANCE_THRESHOLD = 1
            MAX_IOU_THRESHOLD = 0.3

            num_gt = len(VRU_list)
            num_perceived = len(objects['VRU'])
            # Create an N_gt x N_perceived cost matrix
            # Initialize with a very high cost for impossible matches
            cost_matrix = np.full((num_gt, num_perceived), 1000.0)
            # cost_matrix_IoU = np.full((num_gt, num_perceived), 0.0)
            matched_ids_this_cycle = set()
            for i, p in enumerate(VRU_list):
                gt_loc = p.get_location()


                for j, obstacle_ped in enumerate(objects['VRU']):
                    obstacle_loc = obstacle_ped.get_location()
                    compatible = False
                    if obstacle_ped.itsType == 'pedestrian' and 'pedestrian' in p.type_id:
                        compatible = True
                    elif obstacle_ped.itsType != 'pedestrian' and 'pedestrian' not in p.type_id:
                        if p.attributes["number_of_wheels"] == '2':
                            compatible = True

                    if not compatible:
                        continue

                    # squared euclidean distance
                    dx = gt_loc.x - obstacle_loc.x
                    dy = gt_loc.y - obstacle_loc.y
                    distance_sq = dx * dx + dy * dy
                    distance = np.sqrt(distance_sq)
                    current_cost = distance

                    # check proximity
                    if distance <= MAX_DISTANCE_THRESHOLD * 3:
                        cost_matrix[i, j] = current_cost

                    # Todo: consider also IoU for the cost matrix (and mAP calculation)
                    #IoU cost matrix
                    # bbx = p.bounding_box
                    # width = bbx.extent.x*2
                    # length = bbx.extent.y*2
                    # heigth = bbx.extent.z*2
                    # yaw = p.get_transform().rotation.yaw
                    # lidar_transform = self.lidar.sensor.get_transform()
                    # gt_o3d_aabb, gt_obb = self.get_o3d_bounding_boxes(gt_loc.x, gt_loc.y,0, width, length, heigth,
                    #                                                       yaw, lidar_transform)
                    # detected_o3d_aabb, detected_obb = self.get_o3d_bounding_boxes(obstacle_loc.x, obstacle_loc.y, 0,
                    #                                                                   obstacle_ped.bounding_box.extent.x,
                    #                                                                   obstacle_ped.bounding_box.extent.y,
                    #                                                                   obstacle_ped.bounding_box.extent.z,
                    #                                                                   obstacle_ped.yaw,lidar_transform)
                    # IoU = self.compute_obb_iou(gt_obb, detected_obb)
                    # cost_matrix_IoU[i, j] = IoU


            # Perform the optimal assignment using the Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            print(f"[{self.vehicle.get_world().get_snapshot().elapsed_seconds}] ")
            # print("Cost matrix:\n", cost_matrix[:, :])
            # print("Cost matrix IoU:\n", cost_matrix_IoU[:, :])
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            matched_gt_ids_this_cycle = set()
            matched_perceived_ids_this_cycle = set()

            # row_ids contains indices of gt objects
            # col_ind contains indices of perceived matched objects
            for gt_idx, perceived_idx in zip(row_ind, col_ind):
                cost = cost_matrix[gt_idx, perceived_idx]

                is_valid_match = False
                if cost != 1000.0 and cost <= MAX_DISTANCE_THRESHOLD:  # Check distance threshold
                    perceived_obj = objects['VRU'][perceived_idx]
                    gt_obj = VRU_list[gt_idx]
                    is_valid_match = True
                    # print(f"ID: {gt_obj.id} - cost = {cost} - confidence = {perceived_obj.confidence}")

                if is_valid_match:
                    true_positives += 1
                    gt_obj = VRU_list[gt_idx]
                    best_matching_obj = objects['VRU'][perceived_idx]

                    # Update the perceived object with ground truth information
                    actor = allactor.find(gt_obj.id)
                    if actor:
                        aspeed = actor.get_velocity()
                        best_matching_obj.velocity.x = aspeed.x
                        best_matching_obj.velocity.y = aspeed.y

                        atransform = actor.get_transform()
                        yaw = atransform.rotation.yaw
                        # Log heading difference if needed
                        self.log_heading_to_excel(time, gt_obj.id, yaw, best_matching_obj.yaw)

                        best_matching_obj.yaw = yaw

                    best_matching_obj.set_carla_id(gt_obj.id)  # Assign the Carla ID from GT for tracking

                    matched_gt_ids_this_cycle.add(gt_obj.id)
                    matched_perceived_ids_this_cycle.add(best_matching_obj.carla_id)

            # Calculate False Negatives (GT objects not matched)
            # extremely dependent on GT list
            # todo: address false negatives - which GT I SHOULD be detecting? gt_list is not enough
            for gt_obj in VRU_list:
                if gt_obj.id not in matched_gt_ids_this_cycle:
                    false_negatives += 1

            # Calculate False Positives (Perceived objects not matched)
            for perceived_obj in objects['VRU']:
                if perceived_obj.carla_id not in matched_perceived_ids_this_cycle:
                    if perceived_obj.carla_id == -1:
                        false_positives += 1
                    else:  # If it got a Carla ID but wasn't in our current matched list, something went wrong
                        # This assumes all matched perceived_obj will have their carla_id set
                        pass

            # self.log_detection_par(true_positives, false_negatives, false_positives)
            # print(f"True Positives: {true_positives}")
            # print(f"False Negatives: {false_negatives}")
            # print(f"False Positives: {false_positives}")



    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.

        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        if self.ego_pos:
            vehicle_location = self.ego_pos.location
            vehicle_waypoint = self._map.get_waypoint(vehicle_location)

            activate_tl, light_trigger_location = \
                self._get_active_light(tl_list, vehicle_location, vehicle_waypoint)

            objects.update({'traffic_lights': []})

            if activate_tl is not None:
                traffic_light = TrafficLight(activate_tl,
                                             light_trigger_location,
                                             activate_tl.get_state())
                objects['traffic_lights'].append(traffic_light)

        return objects

    def _get_active_light(self, tl_list, vehicle_location, vehicle_waypoint):
        for tl in tl_list:
            object_location = \
                TrafficLight.get_trafficlight_trigger_location(tl)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x +\
                        ve_dir.y * wp_dir.y + \
                        ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue
            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            return tl, object_waypoint.transform.location

        return None, None

    def destroy(self):
        """
        Destroy sensors.
        """
        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()

    # def get_o3d_bounding_boxes(self, x, y, z, extent_x, extent_y, extent_z, heading_deg, lidar_transform):
    #     """
    #     Creates an Open3D Oriented Bounding Box (OBB) and its corresponding
    #     Axis-Aligned Bounding Box (AABB) in the sensor's coordinate frame
    #     by directly transforming the OBB object.
    #
    #     Args:
    #         x, y, z (float): Center of the box in world coordinates.
    #         extent_x, extent_y, extent_z (float): The full size (length, width, height) of the box.
    #         heading_deg (float): The yaw/heading of the box in degrees.
    #         lidar_transform (carla.Transform): The transform of the sensor.
    #
    #     Returns:
    #         tuple: (o3d.geometry.AxisAlignedBoundingBox, o3d.geometry.OrientedBoundingBox)
    #     """
    #     # Use the input parameters to define the OBB directly.
    #     center_world = [x, y, z]
    #     extent_world = [extent_x, extent_y, extent_z]
    #
    #     # Create the rotation matrix from the yaw angle
    #     yaw_rad = np.deg2rad(heading_deg)
    #     R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw_rad))
    #
    #     # Create the Open3D OBB object in world coordinates
    #     obb_world = o3d.geometry.OrientedBoundingBox(center_world, R, extent_world)
    #
    #     # 1. Get the 8 corners of the bounding box
    #     corner_points_world = np.asarray(obb_world.get_box_points())  # shape (8, 3)
    #
    #     # 2. Transpose the array to have shape (3, 8)
    #     corner_points_world_transposed = corner_points_world.transpose()
    #
    #     # 3. Add a row of ones to create homogeneous coordinates, shape (4, 8)
    #     homogeneous_points = np.vstack((corner_points_world_transposed, np.ones(8)))
    #
    #     # Now you can use this as input for your world_to_sensor function
    #     sensor_cords = st.world_to_sensor(homogeneous_points, lidar_transform)
    #
    #     sensor_cords[:1, :] = - sensor_cords[:1, :]
    #     sensor_cords = sensor_cords[:-1, :]
    #     sensor_cords = sensor_cords.transpose()
    #
    #     obb_points = o3d.utility.Vector3dVector(sensor_cords)
    #
    #     # 4. Now, create the AABB and visualization objects from the final,
    #     # correctly transformed OBB, which is the most robust method.
    #     aabb_sensor = o3d.geometry.AxisAlignedBoundingBox.create_from_points(obb_points)
    #     aabb_sensor.color = (0, 1, 0)  # Green for AABB
    #
    #     obb_sensor = o3d.geometry.OrientedBoundingBox.create_from_points(obb_points)
    #     obb2 = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb_sensor)
    #
    #
    #     # This is an alternative to the line_set creation, but it is more efficient
    #     # line_set_sensor = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_sensor)
    #     # line_set_sensor.paint_uniform_color((1, 0, 0)) # Red for OBB outline
    #
    #     # Returning the OBB is critical for your IoU calculation
    #     return aabb_sensor, obb_sensor

    # def compute_obb_iou(self, obb1, obb2):
    #     """
    #     Computes the Intersection over Union (IoU) for two Open3D OrientedBoundingBoxes.
    #
    #     This method works by converting the OBBs to meshes, calculating the
    #     volume of their intersection, and then applying the IoU formula.
    #
    #     Note: Requires Open3D version >= 0.16.0 for mesh boolean operations.
    #     """
    #     try:
    #         mesh1 = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb1)
    #         mesh2 = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(obb2)
    #     except AttributeError:
    #         # Fallback for older versions if needed, though this is less likely
    #         print("Warning: create_from_oriented_bounding_box not found. "
    #               "Check your Open3D version.")
    #         return 0.0
    #
    #     # Get the volumes of the original boxes
    #     volume1 = obb1.volume()
    #     volume2 = obb2.volume()
    #
    #     # Calculate the intersection mesh
    #     # The intersection operation might fail for some geometries, so we use a try-except block
    #     try:
    #         intersection_mesh = mesh1.boolean_intersection(mesh2)
    #         intersection_volume = intersection_mesh.get_volume()
    #     except Exception as e:
    #         # If the boolean operation fails, it often means there's no intersection
    #         # or the intersection is a lower-dimensional shape (line/point).
    #         # print(f"Mesh intersection failed: {e}")
    #         intersection_volume = 0.0
    #
    #     # Calculate the union volume
    #     union_volume = volume1 + volume2 - intersection_volume
    #
    #     # Compute the IoU
    #     if union_volume == 0:
    #         return 0.0  # Avoid division by zero
    #     else:
    #         return intersection_volume / union_volume

    def log_excluded_pedestrian_count(self, perceiver_id_val, count_of_excluded_peds):
        """
        Logs the timestamp, perceiver ID, and the count of excluded pedestrians
        to a CSV file in a thread-safe manner.

        Args:
            perceiver_id_val: The ID of the perceiving entity (e.g., self.id).
            count_of_excluded_peds: The number of pedestrians excluded in this step.
        """
        # timestamp = datetime.datetime.now().isoformat()
        timestamp = time.time_ns()
        data_row = [timestamp, perceiver_id_val, count_of_excluded_peds]

        # Acquire the lock before performing file operations
        with self.excluded_ped_count_csv_lock:
            try:
                file_exists = os.path.isfile(self.EXCLUDED_PED_COUNT_CSV_FILE)
                # Check if file is empty (it might exist but have no content/headers)
                is_empty_file = file_exists and os.path.getsize(self.EXCLUDED_PED_COUNT_CSV_FILE) == 0

                with open(self.EXCLUDED_PED_COUNT_CSV_FILE, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists or is_empty_file:
                        # Write header only if it's a new or empty file
                        writer.writerow(self.EXCLUDED_PED_COUNT_HEADERS)
                    writer.writerow(data_row)
            except IOError as e:
                print(
                    f"Thread ID {threading.get_ident()}: IOError writing to {self.EXCLUDED_PED_COUNT_CSV_FILE}. Reason: {e}")
            except Exception as e:
                print(
                    f"Thread ID {threading.get_ident()}: Unexpected error during CSV writing to {self.EXCLUDED_PED_COUNT_CSV_FILE}. Reason: {e}")

    def log_confidence_level(self, confidence, obj_id):
        """
        Logs the timestamp, perceiver ID, and the confidence level
        to a CSV file in a thread-safe manner.

        Args:
            confidence: confidence level from YOLOv5 detection.
            obj_id: id of the perceived object.
        """
        # timestamp = datetime.datetime.now().isoformat()
        timestamp = time.time_ns()
        data_row = [timestamp, confidence, obj_id]

        # Acquire the lock before performing file operations
        with self.confidence_csv_lock:
            try:
                file_exists = os.path.isfile(self.confidence_csv_file)
                # Check if file is empty (it might exist but have no content/headers)
                is_empty_file = file_exists and os.path.getsize(self.confidence_csv_file) == 0

                with open(self.confidence_csv_file, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists or is_empty_file:
                        # Write header only if it's a new or empty file
                        writer.writerow(self.confidence_csv_header)
                    writer.writerow(data_row)
            except IOError as e:
                print(
                    f"Thread ID {threading.get_ident()}: IOError writing to {self.confidence_csv_file}. Reason: {e}")
            except Exception as e:
                print(
                    f"Thread ID {threading.get_ident()}: Unexpected error during CSV writing to {self.confidence_csv_file}. Reason: {e}")

    def log_heading_to_excel(self, time, actor_id, ground_truth_heading, perceived_heading):

        header = ['timestamp','ground_truth_heading', 'perceived_heading']

        df = pd.DataFrame([{
            'timestamp': time.time_ns(),
            'ground_truth_heading': ground_truth_heading,
            'perceived_heading': perceived_heading
        }])


        log_directory = "Heading_difference"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        file_name = os.path.join(log_directory, f"heading_{actor_id}.csv")

        write_header = not os.path.exists(file_name)

        df.to_csv(file_name, mode='a', header=write_header, index=False)
        # Check if the file exists

    def log_detection_par(self, true_positives, false_negatives, false_positives):
        """
        Logs the detection parameters for mAP calculation
        to a CSV file in a thread-safe manner.

        Args:
          true_positives: detected and matched
          false_negatives: GT objects not matched
          false_positives: Perceived objects not matched)

        """
        # timestamp = datetime.datetime.now().isoformat()
        timestamp = time.time_ns()
        data_row = [true_positives, false_negatives, false_positives]

        # Acquire the lock before performing file operations
        with self.confidence_csv_lock:
            try:
                file_exists = os.path.isfile(self.detection_csv_file)
                # Check if file is empty (it might exist but have no content/headers)
                is_empty_file = file_exists and os.path.getsize(self.detection_csv_file) == 0

                with open(self.detection_csv_file, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists or is_empty_file:
                        # Write header only if it's a new or empty file
                        writer.writerow(self.detection_csv_header)
                    writer.writerow(data_row)
            except IOError as e:
                print(
                    f"Thread ID {threading.get_ident()}: IOError writing to {self.detection_csv_file}. Reason: {e}")
            except Exception as e:
                print(
                    f"Thread ID {threading.get_ident()}: Unexpected error during CSV writing to {self.detection_csv_file}. Reason: {e}")
