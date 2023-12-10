"""
Author: Dikshant Gupta
Time: 22.02.22 21:43
"""

import heapq as hq
import math
import matplotlib.pyplot as plt
import numpy as np
import time
from assets.occupancy_grid import OccupancyGrid


class Node:
    def __init__(self, f=None, g=None, prev=None, prev_d=None, node_c=None, node_d=None, weight=None):
        self.f = f
        self.g = g
        self.f_prime = weight * f
        self.prev = prev
        self.prev_d = prev_d
        self.node_c = node_c
        self.node_d = node_d
        self.cost = self.g + self.f_prime

    def print(self):
        print("Previous: ", self.prev_d)


# total cost f(n) = actual cost g(n) + heuristic cost h(n)
class AnytimeHybridAStar:
    def __init__(self, min_x, max_x, min_y, max_y, obstacle=(), vehicle_length=2.0):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.obstacle = obstacle
        self.vehicle_length = vehicle_length
        # print("Vehicle length: {:.2f}".format(vehicle_length))

        self.obstacles = set(self.obstacle)

    def hgcost(self, position, target, occupancy_grid):
        # Euclidean distance
        output = self.dist(position, target)
        location = [min(round(position[0] - self.min_x), occupancy_grid.shape[0] - 1),
                    min(round(position[1] - self.min_y), occupancy_grid.shape[1] - 1)]
        cost = occupancy_grid[location[0], location[1]]
        return float(output + cost)

    def dist(self, position, target):
        # output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2) +
        #                  (math.radians(position[2]) - math.radians(target[2])) ** 2)
        output = abs(position[0] - target[0]) + abs(position[1] - target[1]) + \
                 abs(math.radians(position[2]) - math.radians(target[2]))
        return float(output) * 1.0

    def next_node(self, location, aph, d):
        theta = math.radians(location[2])
        alpha = math.radians(aph)
        if aph < 0.0001:
            new_x = location[0] + d * math.cos(theta)
            new_y = location[1] + d * math.sin(theta)
            new_theta = theta + (d * math.tan(alpha) / self.vehicle_length)
        else:
            beta = d * math.tan(alpha) / self.vehicle_length
            radii = d / beta
            center_x = location[0] - math.sin(theta) * radii
            center_y = location[1] + math.cos(theta) * radii

            new_x = center_x + math.sin(theta + beta) * radii
            new_y = center_y - math.cos(theta + beta) * radii
            new_theta = theta + beta

        if new_theta > np.pi:
            new_theta = new_theta - (2 * np.pi)

        return new_x, new_y, new_theta

    def find_path(self, start, end, occupancy_grid, agent_locations, speed=1.05, weight=1.0):
        steering_inputs = []
        for i in range(-50, 51, 25):
            steering_inputs.append(i)

        speed_inputs = [min(4.0, max(speed, 1.05))]
        start = (float(start[0]), float(start[1]), float(start[2]))
        end = (float(end[0]), float(end[1]), float(end[2]))
        # The above 2 are in discrete coordinates

        open_heap = []  # element of this list is like (cost,node_d)
        open_diction = {}  # element of this is like node_d:(Node)
        visited_diction = {}  # element of this is like node_d:(Node)

        obstacles = set(agent_locations)
        g = 0

        f = self.hgcost(start, end, occupancy_grid)
        f_prime = weight * f
        hq.heappush(open_heap, (f_prime, start))
        open_diction[start] = Node(f, g, prev=start, prev_d=start, node_c=start, node_d=start, weight=weight)

        incumbent = None
        last_visited_node = None
        upper_bound = 1e5
        t = time.time()

        while len(open_heap) > 0:
            if (time.time() - t) * 1000 > 50.0:
                break
            chosen_d_node = open_heap[0][1]
            current_node = open_diction[chosen_d_node]
            hq.heappop(open_heap)

            if incumbent is None or f < upper_bound:
                visited_diction[chosen_d_node] = open_diction[chosen_d_node]
                last_visited_node = chosen_d_node

                for i in range(len(steering_inputs)):
                    for j in range(len(speed_inputs)):
                        delta = steering_inputs[i]
                        velocity = speed_inputs[j]

                        neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts = self.next_node(current_node.node_c,
                                                                                               delta, velocity)
                        neighbour_theta_cts = math.degrees(neighbour_theta_cts)

                        neighbour_x_d = round(neighbour_x_cts)
                        neighbour_y_d = round(neighbour_y_cts)
                        neighbour_theta_d = round(neighbour_theta_cts)

                        neighbour_d = (neighbour_x_d, neighbour_y_d, neighbour_theta_d)
                        neighbour_cts = (neighbour_x_cts, neighbour_y_cts, neighbour_theta_cts)

                        g = current_node.g + 1.0
                        f = self.hgcost(neighbour_d, end, occupancy_grid)

                        dist = 1000
                        for obs in obstacles:
                            d = np.sqrt((neighbour_x_d - obs[0]) ** 2 + (neighbour_y_d - obs[1]) ** 2)
                            if d < dist:
                                dist = d

                        if g + f < upper_bound and dist > 1.5 and current_node.node_d != neighbour_d:
                            if self.dist(neighbour_d, end) < 3:
                                f = g
                                incumbent = neighbour_d
                                visited_diction[neighbour_d] = Node(0, g, prev=current_node.node_c,
                                                                    prev_d=current_node.node_d, node_c=neighbour_cts,
                                                                    node_d=neighbour_d, weight=weight)
                                upper_bound = f
                            elif neighbour_d in open_diction and open_diction[neighbour_d].g > g:
                                open_diction[neighbour_d] = Node(f, g, current_node.node_c, prev_d=current_node.node_d,
                                                                 node_c=neighbour_cts, node_d=neighbour_d, weight=weight)
                                hq.heappush(open_heap, (open_diction[neighbour_d].f_prime, neighbour_d))
                            elif neighbour_d in visited_diction and visited_diction[neighbour_d].g > g:
                                open_diction[neighbour_d] = Node(f, g, current_node.node_c, prev_d=current_node.node_d,
                                                                 node_c=neighbour_cts, node_d=neighbour_d, weight=weight)
                                visited_diction.pop(neighbour_d)
                            else:
                                open_diction[neighbour_d] = Node(f, g, current_node.node_c, prev_d=current_node.node_d,
                                                                 node_c=neighbour_cts, node_d=neighbour_d, weight=weight)
                                hq.heappush(open_heap, (open_diction[neighbour_d].f_prime, neighbour_d))

        if incumbent is None:
            incumbent = last_visited_node
        paths = []
        node = incumbent
        rev_final_path = [node]  # reverse of final path
        rev_final_path_d = [node]  # reverse of discrete final path
        t = time.time()
        while True:
            if (time.time() - t) * 1000 > 50.0:
                break
            # visited_diction
            open_node_contents = visited_diction[node]  # (Node)
            rev_final_path.append(open_node_contents.prev)
            rev_final_path_d.append(open_node_contents.prev_d)
            node = open_node_contents.prev_d
            if self.dist(node, start) < 3:
                rev_final_path.append(start)
                break
        paths.append(rev_final_path)
        return paths


def main():
    print(__file__ + " start!!")

    # start and goal position
    # (x, y, theta) in meters, meters, degrees
    sx, sy, stheta = 4.067845106124878, 201.1132049560547, -88.49481201171875
    # sx1, sy1, stheta1 = 92, 6, -90
    gx, gy, gtheta = 2.0, 150.0, -90.0

    # sx, sy, stheta = 100, 1, -180
    # gx, gy, gtheta = 70, 1, -180

    # create obstacles
    obstacle = [(-3, 203), (-3, 204), (-3, 205), (-3, 206), (-3, 207), (-3, 208), (-2, 204), (-2, 205), (-2, 206),
                (-2, 207), (-2, 208), (-1, 204), (-1, 205), (-1, 206), (-1, 207), (-1, 208)]

    new_obs = obstacle + [(-2, 203), (-1, 203)]
    # obstacle.append((-1, 209)) # incoming car
    # obstacle = [(85, -2), (85, -1)]
    # obstacle = []
    occupancy_grid = OccupancyGrid()
    hy_a_star = AnytimeHybridAStar(-10, 100, -10, 300, obstacle=[], vehicle_length=4.18)

    g = np.ones((110, 310)) * 1000.0
    sidewalk_cost = 50.0
    road_cost = 1.0
    g[7:13, 13:] = road_cost
    g[97:103, 13:] = road_cost
    g[7:, 7:13] = road_cost
    g[4:7, 4:] = sidewalk_cost
    g[:, 4:7] = sidewalk_cost
    g[13:16, 13:] = sidewalk_cost
    g[94:97, 13:] = sidewalk_cost
    g[103:106, 13:] = sidewalk_cost
    g[13:16, 16:94] = sidewalk_cost

    # K-path estimation with risk
    y = round(sy)
    relaxed_g = g.copy()
    sidewalk_cost = 0.0
    relaxed_g[13:16, y-10:y+10] = sidewalk_cost
    relaxed_g[4:7, y-10:y+10] = sidewalk_cost

    from benchmark.risk.risk_assesment import PerceivedRisk
    risk_estimator = PerceivedRisk()
    cmp = np.ones((110, 310)) * 0.0
    sidewalk_cost = 50.0
    cmp[7:13, 13:] = 1.0
    cmp[97:103, 13:] = 1.0
    cmp[7:, 7:13] = 1.0
    cmp[4:7, 4:] = sidewalk_cost
    cmp[:, 4:7] = sidewalk_cost
    cmp[13:16, 13:] = sidewalk_cost
    cmp[94:97, 13:] = sidewalk_cost
    cmp[103:106, 13:] = sidewalk_cost
    cmp[13:16, 16:94] = sidewalk_cost

    for obs in new_obs:
        cmp[obs[0] + 10, obs[1] + 10] = 1000

    t0 = time.time()
    path = hy_a_star.find_path((sx, sy, stheta), (gx, gy, gtheta), relaxed_g, [], speed=1.0, weight=0.9)
    path = path[0]
    path.reverse()
    print(path)
    steering_angle = (path[2][2] - stheta)
    player = [sx, sy, 35, stheta]
    t = time.time()
    risk, _ = risk_estimator.get_risk(player, steering_angle, cmp)
    t_taken = (time.time() - t0) * 1000
    print(len(path), risk, steering_angle, t_taken)

    cp = occupancy_grid.get_costmap([])
    for path in [path]:
        x, y = list(), list()
        for node in path:
            pixel_coord = occupancy_grid.map.convert_to_pixel(node)
            x.append(pixel_coord[0])
            y.append(pixel_coord[1])
        plt.plot(x, y, "-r")
        # obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[0][0], obstacle[0][1], 0])
        # plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        # obstacle_pixel = occupancy_grid.map.convert_to_pixel([obstacle[1][0], obstacle[1][1], 0])
        # plt.scatter([obstacle_pixel[0]], [obstacle_pixel[1]], c="k")
        plt.imshow(cp, cmap='gray')
        # plt.imshow(cp[x[0]-50:x[0]+50, y[0]-200:y[0]+500], cmap="gray")
        plt.show()


if __name__ == '__main__':
    main()
