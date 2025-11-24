# -*- coding: utf-8 -*-
"""
Utilize scenario manager to manage CARLA simulation construction. This script
is used for carla simulation only, and if you want to manage the Co-simulation,
please use cosim_api.py.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import math
import random
import sys
import json
from random import shuffle
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
import logging

import carla
import numpy as np

from opencda.core.common.vehicle_manager import VehicleManager
from opencda.customize.core.common.vehicle_manager import ExtendedVehicleManager
from opencda.core.application.platooning.platooning_manager import \
    PlatooningManager
from opencda.core.common.rsu_manager import RSUManager
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.utils.customized_map_api import \
    load_customized_world, bcolors


def car_blueprint_filter(blueprint_library, carla_version='0.9.11'):
    """
    Exclude the uncommon vehicles from the default CARLA blueprint library
    (i.e., isetta, carlacola, cybertruck, t2).

    Parameters
    ----------
    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    carla_version : str
        CARLA simulator version, currently support 0.9.11 and 0.9.12. We need
        this as since CARLA 0.9.12 the blueprint name has been changed a lot.

    Returns
    -------
    blueprints : list
        The list of suitable blueprints for vehicles.
    """


    if carla_version == '0.9.11':
        print('old version')
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge_charger.police'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooperst'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes-benz.coupe'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.mustang.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz2017'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    else:
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge.charger_police'),
            blueprint_library.find('vehicle.dodge.charger_police_2020'),
            blueprint_library.find('vehicle.dodge.charger_2020'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooper_s'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes.coupe'),
            blueprint_library.find('vehicle.mercedes.coupe_2020'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.ford.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz_2017'),
            blueprint_library.find('vehicle.lincoln.mkz_2020'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    return blueprints


def multi_class_vehicle_blueprint_filter(label, blueprint_library, bp_meta):
    """
    Get a list of blueprints that have the class equals the specified label.

    Parameters
    ----------
    label : str
        Specified blueprint.

    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    bp_meta : dict
        Dictionary of {blueprint name: blueprint class}.

    Returns
    -------
    blueprints : list
        List of blueprints that have the class equals the specified label.

    """
    blueprints = [
        blueprint_library.find(k)
        for k, v in bp_meta.items() if v["class"] == label
    ]
    return blueprints


class ScenarioManager:
    """
    The manager that controls simulation construction, backgound traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Carla HD Map.

    """

    def __init__(self, scenario_params,
                 apply_ml,
                 carla_version,
                 xodr_path=None,
                 town=None,
                 cav_world=None,
                 carla_host='localhost',
                 carla_port=2000):
        self.scenario_params = scenario_params
        self.carla_version = carla_version

        simulation_config = scenario_params['world']

        # set random seed if stated
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        self.client = \
            carla.Client(carla_host, carla_port)
        self.client.set_timeout(10.0)

        if xodr_path:
            self.world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                self.world = self.client.load_world(town)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % town)
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        self.origin_settings = self.world.get_settings()
        new_settings = self.world.get_settings()

        if simulation_config['sync_mode']:
            new_settings.synchronous_mode = True
            new_settings.fixed_delta_seconds = \
                simulation_config['fixed_delta_seconds']
        else:
            sys.exit(
                'ERROR: Current version only supports sync simulation mode')

        self.world.apply_settings(new_settings)

        # set weather
        weather = self.set_weather(simulation_config['weather'])
        self.world.set_weather(weather)

        # Define probabilities for each type of blueprint
        self.use_multi_class_bp = scenario_params["blueprint"][
            'use_multi_class_bp'] if 'blueprint' in scenario_params else False

        if self.use_multi_class_bp:
            # bbx/blueprint meta
            with open(scenario_params['blueprint']['bp_meta_path']) as f:
                self.bp_meta = json.load(f)
            self.bp_class_sample_prob = scenario_params['blueprint'][
                'bp_class_sample_prob']

            # normalize probability
            self.bp_class_sample_prob = {
                k: v / sum(self.bp_class_sample_prob.values()) for k, v in
                self.bp_class_sample_prob.items()}

        self.cav_world = cav_world
        self.carla_map = self.world.get_map()
        self.apply_ml = apply_ml

    @staticmethod
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather

    def create_vehicle_manager(self, application,
                               map_helper=None,
                               data_dump=False,
                               pldm=False,
                               log_dir=None,port=8000):
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating single CAVs.')
        # By default, we use lincoln as our cav model.
        traffic_config = self.scenario_params['carla_traffic_manager']
        if port != 8000:
            tm = self.client.get_trafficmanager(port)
        else:
            tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)
        single_cav_list = []

        for i, cav_config in enumerate(
                self.scenario_params['scenario']['single_cav_list']):
            # in case the cav wants to join a platoon later
            # it will be empty dictionary for single cav application
            platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base', {})})
            cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                         platoon_base,
                                         cav_config)
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform
            if 'spawn_special' not in cav_config:
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=cav_config['spawn_position'][0],
                        y=cav_config['spawn_position'][1],
                        z=cav_config['spawn_position'][2]),
                    carla.Rotation(
                        pitch=cav_config['spawn_position'][5],
                        yaw=cav_config['spawn_position'][4],
                        roll=cav_config['spawn_position'][3]))
            else:
                # spawn_transform = map_helper(self.carla_version,
                #                             cav_config['spawn_special'][0])
                transform_point = carla.Transform(carla.Location(x=-1202.0827,
                                                                 y=458.2501,
                                                                 z=0.3),
                                                  carla.Rotation(yaw=-20.4866))

                begin_point = carla.Transform(carla.Location(x=-16.7102,
                                                             y=15.3622,
                                                             z=0.3),
                                              carla.Rotation(yaw=-20.4866))

                transform_point.location.x = transform_point.location.x + cav_config['spawn_special'][0] * \
                                             (begin_point.location.x -
                                              transform_point.location.x)
                transform_point.location.y = transform_point.location.y + cav_config['spawn_special'][0] * \
                                             (begin_point.location.y -
                                              transform_point.location.y)
                spawn_transform = transform_point
                # self.world.debug.draw_string(ego_pos.location, str(self.vehicle.id), False, carla.Color(200, 200, 0))



            if 'ms-van3t' in cav_config['v2x']:
                cav_vehicle_bp.set_attribute('color', '0, 0, 255')
                if 'intruder' in cav_config['v2x']:
                    cav_vehicle_bp.set_attribute('color', '255, 0, 0')
                # print ('transform:', spawn_transform)
                vehicle = self.world.spawn_actor(cav_vehicle_bp, spawn_transform)
                # create vehicle manager for each cav
                vehicle_manager = ExtendedVehicleManager(
                    vehicle, cav_config, application,
                    self.carla_map, self.cav_world,
                    current_time=self.scenario_params['current_time'],
                    data_dumping=data_dump,
                    pldm=pldm, log_dir=log_dir, ms_vanet=True)
            else:
                if 'spawn_special' not in cav_config:
                    cav_vehicle_bp.set_attribute('color', '0, 0, 255')
                else:
                    cav_vehicle_bp.set_attribute('color', '255, 0, 0')
                vehicle = self.world.spawn_actor(cav_vehicle_bp, spawn_transform)
                # create vehicle manager for each cav
                vehicle_manager = ExtendedVehicleManager(
                    vehicle, cav_config, application,
                    self.carla_map, self.cav_world,
                    current_time=self.scenario_params['current_time'],
                    data_dumping=data_dump,
                    pldm=pldm, log_dir=log_dir)

            vehicle.set_autopilot(True, tm.get_port())

            if 'vehicle_speed_perc' in cav_config:
                tm.vehicle_percentage_speed_difference(
                    vehicle, cav_config['vehicle_speed_perc'])
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])
            tm.ignore_lights_percentage(vehicle, 0)

            self.world.tick()

            vehicle_manager.v2x_manager.set_platoon(None)

            destination = carla.Location(x=cav_config['destination'][0],
                                         y=cav_config['destination'][1],
                                         z=cav_config['destination'][2])
            vehicle_manager.update_info_LDM()
            vehicle_manager.set_destination(
                vehicle_manager.vehicle.get_location(),
                destination,
                clean=True)


            single_cav_list.append(vehicle_manager)

        return single_cav_list

    def create_vehicle_manager_auto(self, application,
                                           map_helper=None,
                                           data_dump=False,
                                           x=0, y=0, z=0,
                                           number=1,
                                           random_bp=False,
                                           spawnPoint=None):
        single_cav_list = []
        number = len(self.scenario_params['scenario']['single_cav_list'])
        for i in range(number):
            cav_config = self.scenario_params['scenario']['single_cav_list'][i]
            platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base', {})})
            cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                         platoon_base,
                                         cav_config)
            spawn_transform = carla.Transform(
                carla.Location(
                    x=cav_config['spawn_position'][0],
                    y=cav_config['spawn_position'][1],
                    z=cav_config['spawn_position'][2]),
                carla.Rotation(
                    pitch=cav_config['spawn_position'][5],
                    yaw=cav_config['spawn_position'][4],
                    roll=cav_config['spawn_position'][3]))

            cav = self.create_single_vehicle_manager_auto(application, map_helper, data_dump, i, random_bp, number=i,
                                                          spawnPoint=spawn_transform)
            for c in cav:
                single_cav_list.append(c)

        return single_cav_list
        

    def create_single_vehicle_manager_auto(self, application,
                                           map_helper=None,
                                           data_dump=False,
                                           x=0, y=0, z=0,
                                           number=1,
                                           random_bp=False,
                                           spawnPoint=None):
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating single CAV.')
        # By default, we use lincoln as our cav model.
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        if random_bp:
            cav_vehicle_bp = \
                random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
        else:
            cav_vehicle_bp = \
                self.world.get_blueprint_library().find(default_model)
        single_cav_list = []

        # in case the cav wants to join a platoon later
        # it will be empty dictionary for single cav application
        cav_config = self.scenario_params['scenario']['single_cav_list'][number]
        platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base', {})})
        cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                     platoon_base,
                                     cav_config)

        self.world.get_actors()
        actor_list = self.world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        spawnPoints = self.world.get_map().get_spawn_points()

        # iterate over the spawn points and do self.world.debug.draw_string(spawnPoints[i].location, str(round(i)), False, carla.Color(200, 200, 0), 10)
        for i in range(len(spawnPoints)):
            x = spawnPoints[i].location.x
            y = spawnPoints[i].location.y
            text = f"{round(i)} - ({x},{y})"
            self.world.debug.draw_string(spawnPoints[i].location, text, False, carla.Color(200, 200, 0), 10)


        cav_vehicle_bp.set_attribute('color', '0, 255, 0')

        #using spawn point from yaml file.
        # sp= carla.Transform(
        #     carla.Location(
        #         x=cav_config['spawn_position'][0],
        #         y=cav_config['spawn_position'][1],
        #         z=cav_config['spawn_position'][2]),
        #     carla.Rotation(
        #         pitch=cav_config['spawn_position'][5],
        #         yaw=cav_config['spawn_position'][4],
        #         roll=cav_config['spawn_position'][3]))

        vehicle = self.world.spawn_actor(cav_vehicle_bp, spawnPoint)

        # create vehicle manager for each cav
        vehicle_manager = ExtendedVehicleManager(
            vehicle, cav_config, application,
            self.carla_map, self.cav_world,
            current_time=self.scenario_params['current_time'],
            data_dumping=data_dump)

        self.world.tick()

        vehicle_manager.v2x_manager.set_platoon(None)

        destination = carla.Location(x=cav_config['destination'][0],
                                     y=cav_config['destination'][1],
                                     z=cav_config['destination'][2])
        vehicle_manager.update_info_LDM()
        vehicle_manager.set_destination(
            vehicle_manager.vehicle.get_location(),
            destination,
            clean=True)
        vehicle_manager.vehicle.set_autopilot(True, 8000)

        tm = self.client.get_trafficmanager()
        tm.auto_lane_change(vehicle, True)
        tm.ignore_lights_percentage(vehicle_manager.vehicle, 1)

        single_cav_list.append(vehicle_manager)

        return single_cav_list

    def create_single_vehicle_manager(self, application,
                                      map_helper=None,
                                      data_dump=False,
                                      pldm=False,
                                      log_dir=None,
                                      x=0, y=0, z=0):
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating single CAVs.')
        # By default, we use lincoln as our cav model.
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)
        single_cav_list = []

        # in case the cav wants to join a platoon later
        # it will be empty dictionary for single cav application
        cav_config = self.scenario_params['scenario']['single_cav_list'][0]
        platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base', {})})
        cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                     platoon_base,
                                     cav_config)
        # if the spawn position is a single scalar, we need to use map
        # helper to transfer to spawn transform
        if 'spawn_special' not in cav_config:
            spawn_transform = carla.Transform(
                carla.Location(
                    x=x,
                    y=y,
                    z=cav_config['spawn_position'][2]),
                carla.Rotation(
                    pitch=cav_config['spawn_position'][5],
                    yaw=cav_config['spawn_position'][4],
                    roll=cav_config['spawn_position'][3]))
        else:
            spawn_transform = map_helper(self.carla_version,
                                         *cav_config['spawn_special'])

        cav_vehicle_bp.set_attribute('color', '0, 255, 0')
        vehicle = self.world.spawn_actor(cav_vehicle_bp, spawn_transform)

        # create vehicle manager for each cav
        vehicle_manager = ExtendedVehicleManager(
            vehicle, cav_config, application,
            self.carla_map, self.cav_world,
            current_time=self.scenario_params['current_time'],
            data_dumping=data_dump,
            pldm=pldm, log_dir=log_dir)

        self.world.tick()

        vehicle_manager.v2x_manager.set_platoon(None)

        destination = carla.Location(x=cav_config['destination'][0],
                                     y=cav_config['destination'][1],
                                     z=cav_config['destination'][2])
        vehicle_manager.update_info_LDM()
        vehicle_manager.set_destination(
            vehicle_manager.vehicle.get_location(),
            destination,
            clean=True)

        single_cav_list.append(vehicle_manager)

        return single_cav_list

    def create_platoon_manager(self, map_helper=None, data_dump=False):
        """
        Create a list of platoons.

        Parameters
        ----------
        map_helper : function
            A function to help spawn vehicle on a specific position in a
            specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating platoons/')
        platoon_list = []
        self.cav_world = CavWorld(self.apply_ml)

        # we use lincoln as default choice since our UCLA mobility lab use the
        # same car
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)

        # create platoons
        for i, platoon in enumerate(
                self.scenario_params['scenario']['platoon_list']):
            platoon = OmegaConf.merge(self.scenario_params['platoon_base'],
                                      platoon)
            platoon_manager = PlatooningManager(platoon, self.cav_world)
            for j, cav in enumerate(platoon['members']):
                platton_base = OmegaConf.create({'platoon': platoon})
                cav = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                      platton_base,
                                      cav
                                      )
                if 'spawn_special' not in cav:
                    spawn_transform = carla.Transform(
                        carla.Location(
                            x=cav['spawn_position'][0],
                            y=cav['spawn_position'][1],
                            z=cav['spawn_position'][2]),
                        carla.Rotation(
                            pitch=cav['spawn_position'][5],
                            yaw=cav['spawn_position'][4],
                            roll=cav['spawn_position'][3]))
                else:
                    spawn_transform = map_helper(self.carla_version,
                                                 *cav['spawn_special'])

                cav_vehicle_bp.set_attribute('color', '0, 255, 0')
                vehicle = self.world.spawn_actor(cav_vehicle_bp,
                                                 spawn_transform)

                # create vehicle manager for each cav
                vehicle_manager = ExtendedVehicleManager(
                    vehicle, cav, ['platooning'],
                    self.carla_map, self.cav_world,
                    current_time=self.scenario_params['current_time'],
                    data_dumping=data_dump)

                # add the vehicle manager to platoon
                if j == 0:
                    platoon_manager.set_lead(vehicle_manager)
                else:
                    platoon_manager.add_member(vehicle_manager, leader=False)

            self.world.tick()
            destination = carla.Location(x=platoon['destination'][0],
                                         y=platoon['destination'][1],
                                         z=platoon['destination'][2])

            platoon_manager.set_destination(destination)
            platoon_manager.update_member_order()
            platoon_list.append(platoon_manager)

        return platoon_list

    def create_rsu_manager(self, data_dump):
        """
        Create a list of RSU.

        Parameters
        ----------
        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        rsu_list : list
            A list contains all rsu managers..
        """
        print('Creating RSU.')
        rsu_list = []
        for i, rsu_config in enumerate(
                self.scenario_params['scenario']['rsu_list']):
            rsu_config = OmegaConf.merge(self.scenario_params['rsu_base'],
                                         rsu_config)
            rsu_manager = RSUManager(self.world, rsu_config,
                                     self.carla_map,
                                     self.cav_world,
                                     self.scenario_params['current_time'],
                                     data_dump)

            rsu_list.append(rsu_manager)

        return rsu_list


    def spawn_pedestrian_by_list (self, tm, traffic_config, pedestrian_list):
        """
        Spawn the pedestrians by the given list

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        pedestrian_list : list
            The list contains all pedestrians.

        Returns
        -------
        pedestrians_list : list
            Update pedestrians list.
        """

        # list of spawn points for pedestrians near CAV1
        spawn_point_pedestrians = []
        i=0
        while i<400:
            sp = self.world.get_random_location_from_navigation()
            spawn_point_pedestrians.append(sp)
            debug_text = f"ID:{round(i)} X:{sp.x:.1f} Y:{sp.y:.1f}"
            self.world.debug.draw_string(sp, str(round(i)), False, carla.Color(255, 255, 255),
                                            100)
            print(str(round(i)), " : x = ", sp.x, " , y = ", sp.y)
            i=i+1
        spectator = self.world.get_spectator()
        # assign the first CAV as the spectator vehicle
        #spectator_vehicle = single_cav_list[0].vehicle
        # get the transform of the spectator vehicle
        transform = carla.Transform(carla.Location(x=-64,y=24))
        # set the spectator to the top of the spectator vehicle
        spectator.set_transform(carla.Transform(transform.location +
                                                carla.Location(z=110),
                                                carla.Rotation(pitch=-90)))

        blueprint_library = self.world.get_blueprint_library()
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        SpawnActor = carla.command.SpawnActor
        batch = []
        all_id = []
        all_actors = []

        #Create batch with locations to spawn pedestrians
        for i, pedestrian_config in enumerate(traffic_config['pedestrian_list']):
            spawn_transform = carla.Transform(
                carla.Location(
                    x=pedestrian_config['spawn_position'][0],
                    y=pedestrian_config['spawn_position'][1],
                    z=pedestrian_config['spawn_position'][2]))

            walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
            batch.append(SpawnActor(walker_bp, spawn_transform))

            #print spawning points on the map
            self.world.debug.draw_string(spawn_transform.location, str(round(i)), False, carla.Color(200, 0, 200), 100)

        #Spawn pedestrians
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                pedestrian_list.append({"id": results[i].actor_id})

        batch = []
        for i in range(len(pedestrian_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), pedestrian_list[i]['id']))

        #Spawn controllers
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                pedestrian_list[i]['con'] = results[i].actor_id

        for i in range(len(pedestrian_list)):             #pedestrian_list: list of dict {'con': id, 'walker': id}
            all_id.append(pedestrian_list[i]["con"])      #all_id: list of ids [con, walker, con ,walker...]
            all_id.append(pedestrian_list[i]["id"])
        all_actors = self.world.get_actors(all_id)        #all_actors: list of actors [actor con, actor walker, actor con....]


        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())


        print(len(pedestrian_list), "pedestrians were spawned")

        return all_id
    def spawn_pedestrian_by_radius (self, tm, traffic_config):
        """
        Spawn the pedestrians by the given list

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        second_pedestrian_list : list
            The list contains all pedestrians.

        Returns
        -------
        pedestrians_list : list
            Update pedestrians list.
        """

        # list of spawn points for pedestrians near CAV1
        # spawn_point_pedestrians = []
        # i=0
        # while i<150:
        #     sp = self.world.get_random_location_from_navigation()
        #     spawn_point_pedestrians.append(sp)
        #     self.world.debug.draw_string(sp, str(round(i)), False, carla.Color(200, 0, 200),
        #                                     100)
        #     i=i+1
        # spectator = self.world.get_spectator()
        # # assign the first CAV as the spectator vehicle
        # #spectator_vehicle = single_cav_list[0].vehicle
        # # get the transform of the spectator vehicle
        # transform = carla.Transform(carla.Location(x=-64,y=24))
        # # set the spectator to the top of the spectator vehicle
        # spectator.set_transform(carla.Transform(transform.location +
        #                                         carla.Location(z=110),
        #                                         carla.Rotation(pitch=-90)))


        blueprint_library = self.world.get_blueprint_library()
        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        SpawnActor = carla.command.SpawnActor
        batch = []
        all_id = []
        all_actors = []
        spawn_point_pedestrians = []
        i=0
        j=0
        second_pedestrian_list = []
        ped_not_spawned = 0
        n_pedestrian = traffic_config['n_pedestrian']
        radius = traffic_config['radius']
        same_dest = traffic_config['same_destination']

        cav_config = self.scenario_params['scenario']['single_cav_list'][0]
        if traffic_config['center'] == [0, 0]: #if we do not specify, radius is around cav1
            center_x = cav_config['spawn_position'][0]
            center_y = cav_config['spawn_position'][1]
        else:
            center_x = traffic_config['center'][0]
            center_y = traffic_config['center'][1]

        while i < n_pedestrian:
                sp = self.world.get_random_location_from_navigation()
                if (center_x-radius < sp.x < center_x+radius) & (center_y-radius < sp.y < center_y+radius):
                    # spawn_point_pedestrians.append(sp)
                    walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
                    spawnpoint = carla.Transform(carla.Location(sp))
                    batch.append(SpawnActor(walker_bp, spawnpoint))
                    i= i+1
                j=j+1

                if j > 4000:
                    print('Not enough spawn points for the corresponding radius and number of pedestrians chosen.')
                    break


        #Spawn pedestrians
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
                ped_not_spawned = ped_not_spawned + 1
            else:
                second_pedestrian_list.append({"id": results[i].actor_id})

        count = 0
        count2 = 0
        while count < ped_not_spawned:
            sp = self.world.get_random_location_from_navigation()
            if (center_x - radius < sp.x < center_x + radius) & (center_y - radius < sp.y < center_y + radius):
                walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
                spawnpoint = carla.Transform(carla.Location(sp))
                ped = self.world.try_spawn_actor(walker_bp, spawnpoint)
                if ped is not None:
                    second_pedestrian_list.append({"id": ped.id})
                    count = count + 1
                    print("New location found")
            count2 = count2 + 1
            if count2 > 4000:
                print('Not enough spawn points for the corresponding radius and number of pedestrians chosen.')
                break



        batch = []
        for i in range(len(second_pedestrian_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), second_pedestrian_list[i]['id']))

        #Spawn controllers
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                second_pedestrian_list[i]['con'] = results[i].actor_id

        for i in range(len(second_pedestrian_list)):             #pedestrian_list: list of dict {'con': id, 'walker': id}
            all_id.append(second_pedestrian_list[i]["con"])      #all_id: list of ids [con, walker, con ,walker...]
            all_id.append(second_pedestrian_list[i]["id"])
        all_actors = self.world.get_actors(all_id)        #all_actors: list of actors [actor con, actor walker, actor con....]

        if same_dest:
            go_to = self.world.get_random_location_from_navigation()
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(go_to)
        else:
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(self.world.get_random_location_from_navigation())

        print(len(second_pedestrian_list), "pedestrians were spawned by range, center = [ ", center_x, ", ", center_y,
              " ], radius = ", radius)

        return all_id

    def spawn_vehicles_by_list(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given list.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """

        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '0, 255, 0'
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        ego_vehicle_bp = blueprint_library.find(default_model)

        for i, vehicle_config in enumerate(traffic_config['vehicle_list']):
            spawn_transform = carla.Transform(
                carla.Location(
                    x=vehicle_config['spawn_position'][0],
                    y=vehicle_config['spawn_position'][1],
                    z=vehicle_config['spawn_position'][2]),
                carla.Rotation(
                    pitch=vehicle_config['spawn_position'][5],
                    yaw=vehicle_config['spawn_position'][4],
                    roll=vehicle_config['spawn_position'][3]))

            if not traffic_config['random']:
                ego_vehicle_bp.set_attribute('color', color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)

                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            if vehicle_config['blueprint'] == 'bicycle':
                ego_vehicle_bp = blueprint_library.find('vehicle.bh.crossbike')
            elif vehicle_config['blueprint'] == 'motorcycle':
                ego_vehicle_bp = blueprint_library.find('vehicle.harley-davidson.low_rider')
            else: pass

            # spawnPoints = self.world.get_map().get_spawn_points()


            vehicle = self.world.spawn_actor(ego_vehicle_bp, spawn_transform)
            vehicle.set_autopilot(True, tm.get_port())

            if 'vehicle_speed_perc' in vehicle_config:
                tm.vehicle_percentage_speed_difference(
                    vehicle, vehicle_config['vehicle_speed_perc'])
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])
            tm.ignore_lights_percentage(vehicle, 0)

            bg_list.append(vehicle)

        return bg_list

    def spawn_vehicle_by_range(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '128, 128, 128'
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        ego_vehicle_bp = blueprint_library.find(default_model)

        spawn_ranges = traffic_config['range']
        spawn_set = set()
        spawn_num = 0

        for spawn_range in spawn_ranges:
            spawn_num += spawn_range[6]
            x_min, x_max, y_min, y_max = \
                math.floor(spawn_range[0]), math.ceil(spawn_range[1]), \
                    math.floor(spawn_range[2]), math.ceil(spawn_range[3])

            for x in range(x_min, x_max, int(spawn_range[4])):
                for y in range(y_min, y_max, int(spawn_range[5])):
                    location = carla.Location(x=x, y=y, z=0.3)
                    way_point = self.carla_map.get_waypoint(location).transform

                    spawn_set.add((way_point.location.x,
                                   way_point.location.y,
                                   way_point.location.z,
                                   way_point.rotation.roll,
                                   way_point.rotation.yaw,
                                   way_point.rotation.pitch))
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        while count < spawn_num:
            if len(spawn_list) == 0:
                break

            coordinates = spawn_list[0]
            spawn_list.pop(0)

            spawn_transform = carla.Transform(carla.Location(x=coordinates[0],
                                                             y=coordinates[1],
                                                             z=coordinates[
                                                                   2] + 0.3),
                                              carla.Rotation(
                                                  roll=coordinates[3],
                                                  yaw=coordinates[4],
                                                  pitch=coordinates[5]))
            if not traffic_config['random']:
                ego_vehicle_bp.set_attribute('color', color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)
                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            vehicle = \
                self.world.try_spawn_actor(ego_vehicle_bp, spawn_transform)

            if not vehicle:
                continue

            vehicle.set_autopilot(True, tm.get_port())
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            if 'ignore_lights_percentage' in traffic_config:
                tm.ignore_lights_percentage(vehicle,
                                            traffic_config[
                                                'ignore_lights_percentage'])

            # each vehicle have slight different speed
            tm.vehicle_percentage_speed_difference(
                vehicle,
                traffic_config['global_speed_perc'] + random.randint(-30, 30))

            bg_list.append(vehicle)
            count += 1

        return bg_list

    def create_traffic_carla(self, port=8000):
        """
        Create traffic flow.

        Returns
        -------
        tm : carla.traffic_manager
            Carla traffic manager.

        bg_list : list
            The list that contains all the background traffic vehicles.
        """
        print('Spawning CARLA traffic flow.')
        traffic_config = self.scenario_params['carla_traffic_manager']
        if port != 8000:
            tm = self.client.get_trafficmanager(port)
        else:
            tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        bg_list = []

        if isinstance(traffic_config['vehicle_list'], list) or \
                isinstance(traffic_config['vehicle_list'], ListConfig):
            bg_list = self.spawn_vehicles_by_list(tm,
                                                  traffic_config,
                                                  bg_list)

        else:
            bg_list = self.spawn_vehicle_by_range(tm, traffic_config, bg_list)

        pedestrian_list = []
        second_pedestrian_list = []

        if 'pedestrian_list' in traffic_config:
            if isinstance(traffic_config['pedestrian_list'], list) or \
                    isinstance(traffic_config['pedestrian_list'], ListConfig):
                pedestrian_list = self.spawn_pedestrian_by_list(tm,
                                                                traffic_config,
                                                                pedestrian_list)

        #type(traffic_config['pedestrian_by_radius'])
        if 'pedestrian_by_radius' in traffic_config:
            if traffic_config['pedestrian_by_radius'] == True:
                for i, batch_config in enumerate(traffic_config['spawn_list_by_radius']):
                    second_pedestrian_list += self.spawn_pedestrian_by_radius(tm,
                                                                    batch_config)

        all_id = pedestrian_list + second_pedestrian_list
        pedestrian_list = self.world.get_actors(all_id) # list of actors [actor con, actor walker, actor con....]

        print('CARLA traffic flow generated.')
        return tm, bg_list, pedestrian_list

    def create_traffic_carla_by_number(self, vehicle_number):
        traffic_config = self.scenario_params['carla_traffic_manager']
        tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        spawnPoints = self.world.get_map().get_spawn_points()
        for n in range(vehicle_number):
            randInt = random.randint(0, len(spawnPoints) - 1)
            spawn_point = spawnPoints[randInt]
            spawnPoints.pop(randInt)
            bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
            bp.set_attribute('role_name', 'autopilot')
            vehicle = self.world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])
            tm.ignore_lights_percentage(vehicle, 0)

    def create_traffic_carla_by_spawn_point(self, spawnPoints):

        traffic_config = self.scenario_params['carla_traffic_manager']
        tm = self.client.get_trafficmanager()

        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        # iterate over the spawn points and do self.world.debug.draw_string(spawnPoints[i].location, str(round(0)), False, carla.Color(200, 200, 0), 10)
        sps = self.world.get_map().get_spawn_points()
        for i in range(len(sps)):
            self.world.debug.draw_string(sps[i].location, str(round(i)), False, carla.Color(200, 200, 0), 100)

        spawn_points = self.world.get_map().get_spawn_points()
        for sp in spawnPoints:
            spawn_point = spawn_points[sp]
            bp = random.choice(self.world.get_blueprint_library().filter('vehicle.*'))
            bp.set_attribute('role_name', 'autopilot')
            vehicle = self.world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])
            tm.ignore_lights_percentage(vehicle, 0)

    def tick(self):
        """
        Tick the server.
        """
        self.world.tick()

    def destroyActors(self):
        """
        Destroy all actors in the world.
        """

        actor_list = self.world.get_actors()
        for actor in actor_list:
            actor.destroy()

    def close(self):
        """
        Simulation close.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)
