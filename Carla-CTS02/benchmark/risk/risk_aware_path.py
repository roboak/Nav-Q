"""
Author: Dikshant Gupta
Time: 16.01.22 23:11
"""

import numpy as np

from benchmark.path_planner.hybridastar import HybridAStar
from benchmark.path_planner.anytimeastar import AnytimeHybridAStar
from benchmark.risk.risk_assesment import PerceivedRisk


class PathPlanner:
    def __init__(self):
        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.vehicle_length = 4.18
        self.risk_estimator = PerceivedRisk()
        self.path_planner = HybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, [], self.vehicle_length)
        self.anytime_planner = AnytimeHybridAStar(self.min_x, self.max_x, self.min_y, self.max_y, [], self.vehicle_length)

    def find_path(self, start, end, costmap, obstacles, speed, flag):
        if flag:
            paths = self.anytime_planner.find_path(start, end, costmap, obstacles, speed, weight=0.9)
        else:
            paths = self.path_planner.find_path(start, end, costmap, obstacles)
        if len(paths):
            path = paths[0]
        else:
            path = []
        path.reverse()
        return path

    def find_path_with_risk(self, start, end, costmap, obstacles, car_speed, yaw, risk_map, flag, scenario):
        if scenario == 9:
            return self.find_path_with_risk_scenario09(start, end, costmap, obstacles, car_speed, yaw, risk_map, flag)
        if scenario == 11 or False:
            return self.find_path_with_risk_scenario11(start, end, costmap, obstacles, car_speed, yaw, risk_map, flag)
        try:
            path = self.find_path(start, end, costmap, obstacles, car_speed / 3.6, flag)
            if len(path):
                player = [start[0], start[1], car_speed, yaw]
                steering_angle = path[2][2] - start[2]
                risk, drf = self.risk_estimator.get_risk(player, steering_angle, risk_map)
            else:
                risk = np.inf
                # TODO: DRF in this case
        except:
            path, risk = [], np.inf
        return path, risk

    def find_path_with_risk_scenario09(self, start, end, costmap, obstacles, car_speed, yaw, risk_map, flag):
        # checkpoint = (92, 14, -90) original
        checkpoint = (92, 10, -90)
        try:
            if start[1] <= checkpoint[1]:
                if flag:
                    paths = self.anytime_planner.find_path(start, end, costmap, obstacles, car_speed, weight=0.9)
                else:
                    paths = self.path_planner.find_path(start, end, costmap, obstacles)
                if len(paths):
                    path = paths[0]
                else:
                    path = []
                path.reverse()

            else:
                if flag:
                    path_segemnt_1 = self.anytime_planner.find_path(start, checkpoint, costmap, obstacles,
                                                                    car_speed, weight=0.9)[0]
                    path_segemnt_2 = self.anytime_planner.find_path(checkpoint, end, costmap, obstacles,
                                                                    car_speed, weight=0.9)[0]
                    path_segemnt_2.reverse()
                    path_segemnt_1.reverse()
                    path = path_segemnt_1[:-1] + path_segemnt_2[1:]
                else:
                    path_segemnt_1 = self.path_planner.find_path(start, checkpoint, costmap, obstacles)[0]
                    path_segemnt_2 = self.path_planner.find_path(checkpoint, end, costmap, obstacles)[0]
                    path_segemnt_2.reverse()
                    path_segemnt_1.reverse()
                    path = path_segemnt_1[:-1] + path_segemnt_2[1:]

            if len(path):
                player = [start[0], start[1], car_speed, yaw]
                steering_angle = path[2][2] - start[2]
                risk, drf = self.risk_estimator.get_risk(player, steering_angle, risk_map)
            else:
                risk = np.inf
        except:
            path, risk = [], np.inf
        return path, risk

    def find_path_with_risk_scenario11(self, start, end, costmap, obstacles, car_speed, yaw, risk_map, flag):
        checkpoint = (-2, 5, 90)
        try:
            if start[0] <= checkpoint[0]:
                if flag:
                    paths = self.anytime_planner.find_path(start, end, costmap, obstacles, car_speed, weight=0.9)
                else:
                    paths = self.path_planner.find_path(start, end, costmap, obstacles)
                if len(paths):
                    path = paths[0]
                else:
                    path = []
                path.reverse()

            else:
                if flag:
                    path_segemnt_1 = self.anytime_planner.find_path(start, checkpoint, costmap, obstacles,
                                                                    car_speed, weight=0.9)[0]
                    path_segemnt_2 = self.anytime_planner.find_path(checkpoint, end, costmap, obstacles,
                                                                    car_speed, weight=0.9)[0]
                    path_segemnt_2.reverse()
                    path_segemnt_1.reverse()
                    path = path_segemnt_1[:-1] + path_segemnt_2[1:]
                else:
                    path_segemnt_1 = self.path_planner.find_path(start, checkpoint, costmap, obstacles)[0]
                    path_segemnt_2 = self.path_planner.find_path(checkpoint, end, costmap, obstacles)[0]
                    path_segemnt_2.reverse()
                    path_segemnt_1.reverse()
                    path = path_segemnt_1[:-1] + path_segemnt_2[1:]

            if len(path):
                player = [start[0], start[1], car_speed, yaw]
                steering_angle = path[2][2] - start[2]
                risk, drf = self.risk_estimator.get_risk(player, steering_angle, risk_map)
            else:
                risk = np.inf
        except:
            path, risk = [], np.inf
        return path, risk
