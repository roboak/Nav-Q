"""
Author: Dikshant Gupta
Time: 12.07.21 11:03
"""

import carla
import datetime
import math
import time

import cv2
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from collections import deque

from PIL import Image

from benchmark.agent import Agent
from config import Config
from assets.occupancy_grid import OccupancyGrid
from benchmark.path_planner.hybridastar import HybridAStar
from benchmark.risk.risk_aware_path import PathPlanner
# from ped_path_predictor.m2p3 import PathPredictor


class RLAgent(Agent):
    """Base class for HyLEAR Agent"""

    def __init__(self, world, carla_map, scenario):
        super(RLAgent, self).__init__(world.player)
        self.world = world
        self.vehicle = world.player
        self.wmap = carla_map
        self.scenario = scenario
        self.occupancy_grid = OccupancyGrid()
        self.fig = plt.figure()
        self.display_costmap = False
        self.prev_action = None
        self.prev_speed = None
        self.folder = datetime.datetime.now().timestamp()
        self.ped_history = deque(list(), maxlen=15)
        self.past_trajectory = list()
        self.pedestrian_observable = False
        # os.mkdir("_out/{}".format(self.folder))

        obstacle = []
        self.vehicle_length = self.vehicle.bounding_box.extent.x * 2
        self.vehicle_width = self.vehicle.bounding_box.extent.y * 2

        self.grid_cost = np.ones((110, 310)) * 1000.0
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0
        # Sidewalk Network
        self.grid_cost[4:7, 4:] = 50.0
        self.grid_cost[:, 4:7] = 50.0
        self.grid_cost[13:16, 13:] = 50.0
        self.grid_cost[94:97, 13:] = 50.0
        self.grid_cost[103:106, 13:] = 50.0
        self.grid_cost[13:16, 16:94] = 50.0

        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, obstacle, self.vehicle_length)

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

        self.risk_path_planner = PathPlanner()
        # self.ped_pred = PathPredictor("ped_path_predictor/_out/m2p3_289271.pth")
        # self.ped_pred.model.eval()

    def update_scenario(self, scenario):
        self.scenario = scenario
        self.ped_history = deque(list(), maxlen=15)
        self.past_trajectory = list()
        self.pedestrian_observable = False

    # check if an obstacle is within close enough
    def in_rectangle(self, x, y, theta, ped_x, ped_y, front_margin=1.5, side_margin=0.5, back_margin=0.5, debug=False):
        theta = theta / (2 * np.pi)
        # TOP RIGHT VERTEX:
        top_right_x = x + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                      ((front_margin + self.vehicle_length / 2) * np.cos(theta))
        top_right_y = y - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                      ((front_margin + self.vehicle_length / 2) * np.sin(theta))

        # TOP LEFT VERTEX:
        top_left_x = x - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) + \
                     ((front_margin + self.vehicle_length / 2) * np.cos(theta))
        top_left_y = y + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) + \
                     ((front_margin + self.vehicle_length / 2) * np.sin(theta))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x - ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                     ((back_margin + self.vehicle_length / 2) * np.cos(theta))
        bot_left_y = y + ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                     ((back_margin + self.vehicle_length / 2) * np.sin(theta))

        # BOTTOM RIGHT VERTEX:
        bot_right_x = x + ((side_margin + self.vehicle_width / 2) * np.sin(theta)) - \
                      ((back_margin + self.vehicle_length / 2) * np.cos(theta))
        bot_right_y = y - ((side_margin + self.vehicle_width / 2) * np.cos(theta)) - \
                      ((back_margin + self.vehicle_length / 2) * np.sin(theta))

        if debug:
            print("Top Left ", top_left_x, top_left_y)
            print("Top Right ", top_right_x, top_right_y)
            print("Bot Left ", bot_left_x, bot_left_y)
            print("Bot Right ", bot_right_x, bot_right_y)

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]
        return 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

    def linmap(self, a, b, c, d, x):
        return (x - a) / (b - a) * (d - c) + c

    def get_reward(self, action):
        reward = 0
        goal = False
        terminal = False

        velocity = self.vehicle.get_velocity()
        speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6  # in kmph
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        end = self.scenario[2]
        goal_dist = np.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)

        if speed > 1.0:
            if speed <= 20:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=1, side_margin=0.5)
            else:
                ped_hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                            front_margin=2, side_margin=0.5)
            if ped_hit:
                # scale penalty by impact speed
                # hit = True
                scaling = self.linmap(0, Config.max_speed, 0, 1, min(speed, Config.max_speed))  # in kmph
                # Different penal;ties for near miss and actual collisision.
                if self.world.collision_sensor.flag:
                    collision_reward = Config.hit_penalty * (scaling + 0.1)
                else:
                    collision_reward = Config.near_miss_penalty * (scaling + 0.1)
                # if collision_reward >= 700:
                #     terminal = True
                reward -= collision_reward

        #TODO: How are these numbers calculated?
        # reward = -goal_dist / 1000
        reward -= pow(goal_dist / 4935.0, 0.8) * 1.2

        # All grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            xmin = round(car_x - self.vehicle_width / 2)
            xmax = round(car_x + self.vehicle_width / 2)
            ymin = round(car_y - self.vehicle_length / 2)
            ymax = round(car_y + self.vehicle_length / 2)
            for x in range(xmin, xmax):
                for y in range(ymin, ymax):
                    grid[round(x), round(y)] = 100
            # print(xmin, xmax, ymin, ymax)
            # x = self.world.incoming_car.get_location().x
            # y = self.world.incoming_car.get_location().y
            # grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        location = [min(round(start[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(start[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # "Heavily" penalize braking if you are already standing still
        if self.prev_speed is not None:
            if action != 0 and self.prev_speed < 0.28: # speed less than 1kmph
                reward -= Config.braking_penalty

        # Limit max speed to 50 kmph
        if self.prev_speed is not None:
            if self.prev_speed*3.6 > Config.max_speed:
                reward -= Config.over_speeding_penalty

        # Penalize braking/acceleration actions to get a smoother ride
        if self.prev_action.brake > 0: last_action = 2
        elif self.prev_action.throttle > 0: last_action = 0
        else: last_action = 1
        if last_action != 1 and last_action != action:
            reward -= 0.05

        reward -= pow(abs(self.prev_action.steer), 1.3) / 2.0

        if goal_dist < 3:
            reward += Config.goal_reward
            goal = True
            terminal = True

        # Normalize reward
        reward = reward / 1000.0

        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.2, side_margin=0.2, back_margin=0.1) or obstacle_cost > 50.0
        # hit = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
        #                         front_margin=0.01, side_margin=0.01, back_margin=0.01) or obstacle_cost > 50.0
        hit = self.world.collision_sensor.flag or obstacle_cost > 50.0
        nearmiss = self.in_rectangle(start[0], start[1], start[2], walker_x, walker_y,
                                     front_margin=1.5, side_margin=0.5, back_margin=0.5)
        # if reward<-1:
        #     reward = -1
        return reward, goal, hit, nearmiss, terminal


    def get_local_coordinates(self, path):
        world_to_camera = np.array(self.world.semseg_sensor.sensor.get_transform().get_inverse_matrix())
        world_points = np.array(path)
        world_points = world_points[:, :2]
        world_points = np.c_[world_points, np.zeros(world_points.shape[0]), np.ones(world_points.shape[0])].T
        sensor_points = np.dot(world_to_camera, world_points)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        image_w = int(Config.segcam_image_x)
        image_h = int(Config.segcam_image_y)
        fov = float(Config.segcam_fov)
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0
        points_2d = np.dot(K, point_in_camera_coords)
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        points_2d = points_2d.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        u_coord = points_2d[:, 0].astype(np.intc)
        v_coord = points_2d[:, 1].astype(np.intc)
        return u_coord, v_coord

    def get_car_intention(self, obstacles, path, start):
        self.past_trajectory.append(start)
        # print("debug: ", self.world.semseg_sensor.array)
        while(self.world.semseg_sensor.array is None):
            # print("debug: waiting")
            time.sleep(1)
        car_intention = self.world.semseg_sensor.array.copy()
        if len(path) == 0:
            return car_intention
        x, y = self.get_local_coordinates(path)
        car_intention[y, x, :] = 255.0  # overlay planned path on input with white line

        x, y = self.get_local_coordinates(self.past_trajectory)
        car_intention[y, x, :] = 0.0  # overlay past trajectory on input with black line
        # with open("_out/costmap_{}.pkl".format(start[1]), "wb") as file:
        #     pkl.dump(car_intention, file)
        # car_intention = np.transpose(car_intention, (2, 0, 1))
        # assert car_intention.shape[0] == 3

        # Reducing the dimension oif car intention from 400x400x3 -> 230x100x3
        return car_intention

    def run_step(self, debug=False):
        self.vehicle = self.world.player
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.scenario[2]

        obstacles = self.get_obstacles(start)
        path = self.find_path(start, end, self.grid_cost, obstacles)

        if self.display_costmap:
            self.plot_costmap(obstacles, path)

        control = carla.VehicleControl()
        control.brake = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        if len(path) == 0:
            control.steer = 0
        else:
            target_yaw = path[2][2]
            yaw = start[2]
            delta = (180 - abs(target_yaw)) + (180 - abs(yaw))
            if target_yaw < 0 and yaw > 0:
                control.steer = delta / 70.
            elif target_yaw > 0 and yaw < 0:
                control.steer = -delta / 70.
            else:
                control.steer = (path[2][2] - start[2]) / 70.

        self.prev_action = control
        velocity = self.vehicle.get_velocity()
        self.prev_speed = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5)
        risk = 0.0
        return control, self.get_car_intention(obstacles, path, start), risk

    def get_path_simple(self, start, end, obstacles):
        car_velocity = self.vehicle.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2) * 3.6
        yaw = start[2]

        updated_risk_cmp = np.copy(self.risk_cmp)
        for pos in obstacles:
            pos = (round(pos[0]), round(pos[1]))
            updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000
        if len(self.ped_history) >= 15:
            ped_path = np.array(self.ped_history)
            ped_path = ped_path.reshape((15, 2))
            pedestrian_path = self.ped_pred.get_single_prediction(ped_path)
            for node in pedestrian_path:
                updated_risk_cmp[round(node[0]), round(node[1])] = 10000
        if self.scenario[0] == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            x, y = round(self.world.incoming_car.get_location().x), round(self.world.incoming_car.get_location().y)
            # Hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # All grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))

        if self.scenario[0] in [10, 1] and self.world.walker.get_location().y > start[1] and start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])
        path = self.risk_path_planner.find_path_with_risk(start, end, self.grid_cost, obstacles, car_speed,
                                                          yaw, updated_risk_cmp, True, self.scenario[0])
        # path = self.find_path(start, end, self.grid_cost, obstacles)
        intention = self.get_car_intention([], path[0], start)
        return path, intention

    def find_path(self, start, end, costmap, obstacles):
        checkpoint = (92, 14, -90)
        if self.scenario[0] != 9 or start[1] <= checkpoint[1]:
            t = time.time()
            paths = self.path_planner.find_path(start, end, costmap, obstacles)
            if len(paths):
                path = paths[0]
            else:
                path = []
            path.reverse()
        else:
            path_segemnt_1 = self.path_planner.find_path(start, checkpoint, costmap, obstacles)[0]
            path_segemnt_2 = self.path_planner.find_path(checkpoint, end, costmap, obstacles)[0]
            path_segemnt_2.reverse()
            path_segemnt_1.reverse()
            path = path_segemnt_1[:-1] + path_segemnt_2[1:]
        return path

    def plot_costmap(self, obstacles, path):
        cp = self.occupancy_grid.get_costmap([])
        x, y = list(), list()
        for node in path:
            pixel_coord = self.occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0])
            y.append(pixel_coord[1])
        plt.plot(x, y, "-r")
        if len(obstacles) > 0:
            obstacle_pixel = self.occupancy_grid.map.convert_to_pixel([obstacles[0][0], obstacles[0][1], 0])
            plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        # print(obstacle_pixel)
        plt.imshow(cp, cmap="gray")
        plt.axis([0, 350, 950, 1550])
        plt.draw()
        plt.pause(0.1)
        self.fig.clear()

    def get_obstacles(self, start):
        obstacles = list()
        walker_x, walker_y = self.world.walker.get_location().x, self.world.walker.get_location().y
        walker_flag = False
        if self.scenario[0] == 6:
            if walker_y > start[1]:
                walker_flag = True
        elif walker_y < start[1]:
            walker_flag = True
        if np.sqrt((start[0] - walker_x) ** 2 + (start[1] - walker_y) ** 2) <= 50.0 and walker_flag:
            self.ped_history.append([walker_x, walker_y])
            if self.scenario[0] == 3 and walker_x >= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
                self.pedestrian_observable = True
            elif self.scenario[0] in [7, 8] and walker_x <= self.world.incoming_car.get_location().x:
                obstacles.append((int(walker_x), int(walker_y)))
                self.pedestrian_observable = True
            elif self.scenario[0] in [1, 2, 4, 5, 6, 9, 10]:
                obstacles.append((int(walker_x), int(walker_y)))
                self.pedestrian_observable = True
        if not walker_flag:
            self.pedestrian_observable = False
        if self.scenario[0] in [3, 7, 8, 10]:
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                buffer = 0
                xmin = math.ceil(car_x - self.vehicle_width / 2) - buffer
                xmax = math.ceil(car_x + self.vehicle_width / 2) + buffer
                ymin = math.ceil(car_y - self.vehicle_length / 2) - buffer
                ymax = math.ceil(car_y + self.vehicle_length / 2) + buffer
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))
        if self.scenario[0] == 11:
            self.pedestrian_observable = False
            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                buffer = 0
                xmin = math.ceil(car_x - self.vehicle_width / 2) - buffer
                xmax = math.ceil(car_x + self.vehicle_width / 2) + buffer
                ymin = math.ceil(car_y - self.vehicle_length / 2) - buffer
                ymax = math.ceil(car_y + self.vehicle_length / 2) + buffer
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))
            parked_cars = self.world.parked_cars
            if parked_cars is not None:
                for car in parked_cars:
                    car_x, car_y = car.get_location().x, self.world.incoming_car.get_location().y
                    if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                        buffer = 0
                        xmin = math.ceil(car_x - self.vehicle_width / 2) - buffer
                        xmax = math.ceil(car_x + self.vehicle_width / 2) + buffer
                        ymin = math.ceil(car_y - self.vehicle_length / 2) - buffer
                        ymax = math.ceil(car_y + self.vehicle_length / 2) + buffer
                        for x in range(xmin, xmax + 1):
                            for y in range(ymin, ymax + 1):
                                obstacles.append((int(x), int(y)))
        if self.scenario[0] == 12:
            self.pedestrian_observable = False
            parked_car = self.world.parked_cars[0]
            px, py = round(parked_car.get_location().x), round(parked_car.get_location().y)
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((px + i, py + j))

            car_x, car_y = self.world.incoming_car.get_location().x, self.world.incoming_car.get_location().y
            if np.sqrt((start[0] - car_x) ** 2 + (start[1] - car_y) ** 2) <= 50.0:
                for i in [-1, 0, 1]:
                    for j in [-2, -1, 0, 1, 2]:
                        obstacles.append((car_x + i, car_y + j))
        return obstacles
