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


def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)

        cav_world = CavWorld(opt.apply_ml)

        if 'name' in scenario_params['scenario']['town']:
            town = scenario_params['scenario']['town']['name']
        else:
            print('No town name has been specified, please check the yaml file.')
            raise ValueError

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town=town,
                                                   carla_host=opt.host,
                                                   carla_port=opt.port,
                                                   cav_world=cav_world)

        # create CAVs (vehicles equipped with sensors)
        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list, all_actors = \
            scenario_manager.create_traffic_carla()

        # get spectactor
        spectator = scenario_manager.world.get_spectator()
        # assign the first CAV as the spectator vehicle
        spectator_vehicle = single_cav_list[0].vehicle
        # get the transform of the spectator vehicle
        transform = spectator_vehicle.get_transform()
        # set the spectator to the top of the spectator vehicle
        spectator.set_transform(carla.Transform(transform.location +
                                                carla.Location(z=60),
                                                carla.Rotation(pitch=-90)))
        scenario_manager.tick()

        location = single_cav_list[0].vehicle.get_location()
        print("CAV spawned location:", location)

        # Simulation loop
        while True:
            scenario_manager.tick()
            # update spectator position
            transform = spectator_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location +
                                                    carla.Location(z=60),
                                                    carla.Rotation(pitch=-90)))

            for i, single_cav in enumerate(single_cav_list):
                # update perception and localization info for CAV
                single_cav.update_info_LDM()
                # Vehicle manager set with autopilot --> no control needed
                # control = single_cav.run_step()
                # single_cav.vehicle.apply_control(control)

    finally:
        print("Simulation finished.")
        # for v in single_cav_list:
        #     v.destroy()
        for i in range(0, len(all_actors), 2):
            all_actors[i].stop()

        print(f'\nDestroying {int(len(all_actors)/2)} walkers and {int(len(single_cav_list))} CAVs')
        scenario_manager.destroyActors()

        scenario_manager.close()
