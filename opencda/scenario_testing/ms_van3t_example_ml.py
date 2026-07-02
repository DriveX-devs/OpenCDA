# -*- coding: utf-8 -*-
import os

import carla

import opencda.scenario_testing.utils.cosim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time
from threading import Event
from opencda.scenario_testing.utils.ms_van3t_cosim_api import MsVan3tCoScenarioManager


def _safe_path_part(value):
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(value))


def _top_view_transform(vehicle):
    transform = vehicle.get_transform()
    return carla.Transform(transform.location + carla.Location(z=60),
                           carla.Rotation(pitch=-90))


def run_scenario(opt, scenario_params):
    top_view_camera = None
    try:
        scenario_params = add_current_time(scenario_params)
        frame_dump_dir = os.environ.get('OPENCDA_FRAME_DUMP_DIR')
        if not frame_dump_dir:
            frame_dump_dir = os.path.join(
                os.getcwd(),
                'visualization_frames',
                'ms_van3t_example_ml_%s' %
                _safe_path_part(scenario_params['current_time']))
            os.environ['OPENCDA_FRAME_DUMP_DIR'] = frame_dump_dir
        os.makedirs(frame_dump_dir, exist_ok=True)
        print('Saving OpenCDA visualization frames to %s' % frame_dump_dir)
        carla_top_view_dir = os.path.join(frame_dump_dir, 'Carla_top_view')
        os.makedirs(carla_top_view_dir, exist_ok=True)

        cav_world = CavWorld(opt.apply_ml)

        if 'name' in scenario_params['scenario']['town']:
            town = scenario_params['scenario']['town']['name']
        else:
            print('No town name has been specified, please check the yaml file.')
            raise ValueError

        # create co-simulation scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town=town,
                                                   cav_world=cav_world,
                                                   carla_host=opt.host,
                                                   carla_port=opt.port)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla(port=opt.tm_port)

        step_event = Event()
        stop_event = Event()
        ms_van3t_manager = \
            MsVan3tCoScenarioManager(scenario_params,
                                     scenario_manager,
                                     single_cav_list,
                                     traffic_manager,
                                     step_event=step_event,
                                     stop_event=stop_event)

        spectator = scenario_manager.world.get_spectator()
        spectator_vehicle = single_cav_list[-1].vehicle
        top_view_transform = _top_view_transform(spectator_vehicle)
        spectator.set_transform(top_view_transform)

        camera_bp = scenario_manager.world.get_blueprint_library().find(
            'sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        top_view_camera = scenario_manager.world.spawn_actor(
            camera_bp,
            top_view_transform)
        top_view_frame = [0]

        def save_top_view(image):
            image.save_to_disk(
                os.path.join(carla_top_view_dir,
                             '%06d.png' % top_view_frame[0]))
            top_view_frame[0] += 1

        top_view_camera.listen(save_top_view)
        scenario_manager.tick()

        while True:

            top_view_transform = _top_view_transform(spectator_vehicle)
            spectator.set_transform(top_view_transform)
            top_view_camera.set_transform(top_view_transform)

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info_LDM()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)

            for actor in scenario_manager.world.get_actors().filter("*vehicle*"):
                location = actor.get_location()
                scenario_manager.world.debug.draw_string(location, str(actor.id), False, carla.Color(200, 200, 0))

            step_event.set()
            ms_van3t_manager.carla_object.tick_event.wait()
            ms_van3t_manager.carla_object.tick_event.clear()

    except Exception as e:
        print("Exception detected during the simulation: %s" % str(e))

    finally:
        if top_view_camera is not None:
            top_view_camera.stop()
            top_view_camera.destroy()
        stop_event.set() # stop the co-simulation
        step_event.set() # stop the co-simulation
        scenario_manager.close()
        print("Simulation finished.")
        for v in single_cav_list:
            v.destroy()
