"""
Author: Dikshant Gupta
Time: 11.11.21 15:48
"""

import gym
import numpy as np

# import glob
# import os
# import sys
# try:
#     sys.path.append(glob.glob('/home/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import carla
import pygame
import random
from PIL import Image

from benchmark.environment.world import World
from benchmark.environment.hud import HUD
from benchmark.learner_example import Learner
from benchmark.rlagent import RLAgent
from benchmark.rlagent_mod import RLAgent_mod

from config import Config
from benchmark.scenarios.scenario import Scenario


class GIDASBenchmark(gym.Env):
    def __init__(self, dim_red=False, port=Config.port, mode="TRAINING", record=False):
        super(GIDASBenchmark, self).__init__()
        random.seed(100)
        self.action_space = gym.spaces.Discrete(Config.N_DISCRETE_ACTIONS)
        height = int(Config.segcam_image_x)
        width = int(Config.segcam_image_y)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.dim_red = dim_red
        pygame.init()
        pygame.font.init()

        self.client = carla.Client(Config.host, port)
        self.client.set_timeout(300.0)

        self.control = None
        self.display = None
        self.scenario = None
        self.speed = None
        self.distance = None
        self.record = record
        self.mode = mode
        # self.setting = mode
        self._max_episode_steps = 500
        self.clock = pygame.time.Clock()

        hud = HUD(Config.width, Config.height)
        self.client.load_world('Town01')
        wld = self.client.get_world()
        wld.unload_map_layer(carla.MapLayer.StreetLights)
        wld.unload_map_layer(carla.MapLayer.Props)
        # wld.unload_map_layer(carla.MapLayer.Particles)
        self.map = wld.get_map()
        settings = wld.get_settings()
        settings.fixed_delta_seconds = Config.simulation_step
        settings.synchronous_mode = Config.synchronous
        wld.apply_settings(settings)

        self.scene_generator = Scenario(wld)
        self.scene = self.scene_generator.scenario01()
        self.world = World(wld, hud, self.scene, Config)

        # Chosing RLAgent which returns original car intention of red_dim = False else
        # chosing agent which returns cropped car_intention
        agent = RLAgent_mod if self.dim_red else RLAgent
        # self.planner_agent = RLAgent(self.world, self.map, self.scene)
        self.planner_agent = agent(self.world, self.map, self.scene)

        wld_map = wld.get_map()
        print(wld_map.name)
        wld.tick()

        self.test_episodes_iter = None
        self.test_episodes = list()
        self.episodes = list()
        print(Config.scenarios)
        # mode is set to testing only setting = "special".

        self._get_scenes_for_partial_testing()
        self._get_scenes_for_training()
        self.test_episodes_iter = iter(self.test_episodes)


    def _get_scenes_for_training(self):
        for scenario in Config.scenarios:
            for speed in np.arange(Config.ped_speed_range[0], Config.ped_speed_range[1] + 0.1, 0.1):
                for distance in np.arange(Config.ped_distance_range[0], Config.ped_distance_range[1] + 1, 1):
                    self.episodes.append((scenario, speed, distance))

    def _get_scenes_for_partial_testing(self):
        sample_ped_speeds = np.random.uniform(Config.val_ped_speed_range[0][0], Config.val_ped_speed_range[0][1] + 0.1, 3)
        sample_ped_dists = np.random.uniform(Config.val_ped_distance_range[0], Config.ped_distance_range[1] + 1, 3)
        for scenario in Config.val_scenarios:
            for speed in sample_ped_speeds:
                for distance in sample_ped_dists:
                    self.test_episodes.append((scenario, speed, distance))

    def reset(self):
        scenario_id, ped_speed, ped_distance = self.next_scene()
        # ped_speed = 1.25  # Debug Settings
        # ped_distance = 10.75
        # scenario_id = "10"
        self.scenario = scenario_id
        self.speed = ped_speed
        self.distance = ped_distance
        func = 'self.scene_generator.scenario' + scenario_id
        scenario = eval(func + '()')
        self.world.restart(scenario, ped_speed, ped_distance)
        self.planner_agent.update_scenario(scenario)
        self.client.get_world().tick()
        observation, risk = self._get_observation()
        return observation

    def _get_observation(self):
        # control, observation, risk, ped_observable = self.planner_agent.run_step()
        control, observation, risk = self.planner_agent.run_step()
        # the control received here already has the steering values stored in it.
        self.control = control
        return observation, risk

    def step(self, action):
        self.world.tick(self.clock)
        # velocity of the ego vehicle
        velocity = self.world.player.get_velocity()
        speed = (velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5
        speed *= 3.6
        if self.scenario == '10':
            # Maintain a minimum speed of 20kmph
            if speed < 20:
                action = 0
            elif speed > 50:
                action = 2
        if self.scenario == '11' or self.scenario == '12':
            # Maintain a maximum speed of 20kmph
            if speed > 20:
                action = 2

        if action == 0:
            self.control.throttle = 0.6
        elif action == 2:
            self.control.brake = 0.6
        elif action == 1:
            self.control.throttle = 0
        self.world.player.apply_control(self.control)
        if Config.synchronous:
            frame_num = self.client.get_world().tick()
            if self.record:
                im = Image.fromarray(self.world.camera_manager.array.copy())
                im.save("_out/recordings/frame_{:03d}.png".format(frame_num))

        observation, risk = self._get_observation()
        reward, goal, accident, near_miss, terminal = self.planner_agent.get_reward(action)
        info = {"goal": goal, "accident": accident, "near miss": near_miss,
                "velocity": self.planner_agent.vehicle.get_velocity(), "risk": risk,
                "scenario": self.scenario, "ped_speed": self.speed, "ped_distance": self.distance, "scenario_id": self.scenario}

        if self.mode == "TESTING":
            terminal = goal
        return observation, reward, terminal, info

    def render(self, mode="human"):
        if self.display is None:
            self.display = pygame.display.set_mode(
                (Config.width, Config.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0, 0, 0))
            pygame.display.flip()
        self.world.render(self.display)
        pygame.display.flip()

    def close(self):
        self.world.destroy()
        pygame.quit()

    def reset_agent(self, agent):
        self.planner_agent = agent

    def eval(self, current_episode=0):
        self.mode = "TESTING"
        episodes = list()
        for scenario in Config.test_scenarios:
            if scenario in ['11', '12']:
                for speed in np.arange(10, 20 + 0.1, 0.1):
                    episodes.append((scenario, speed, 0))
            else:
                for speed in np.arange(Config.test_ped_speed_range[0], Config.test_ped_speed_range[1] + 0.1, 0.1):
                    for distance in np.arange(Config.test_ped_distance_range[0], Config.test_ped_distance_range[1] + 1, 1):
                        episodes.append((scenario, speed, distance))

        # episodes.append(('04', 1.8, 34))
        self.episodes = episodes[current_episode:]
        # random.shuffle(episodes)
        print("Total Episodes: ", len(self.episodes))
        self.test_episodes_iter = iter(episodes[current_episode:])

    def next_scene(self):
        if self.mode == "TRAINING":
            return random.choice(self.episodes)
        elif self.mode == "TESTING":
            scene_config = next(self.test_episodes_iter)
            return scene_config
        elif self.mode == "PARTIAL_TESTING":
            scene_config = next(self.test_episodes_iter)
            return scene_config
