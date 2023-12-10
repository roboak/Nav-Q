"""
Author: Dikshant Gupta
Time: 03.08.21 13:22
"""

import numpy as np
import matplotlib.pyplot as plt
from assets.map import CarlaMap


class OccupancyGrid:
    def __init__(self, map_name="Town01"):
        self.map = CarlaMap(map_name)
        self.ref_map = self.map.get_map()
        # extract green channel, invert, scale to range 0..100, convert to int8
        self.ref = (self.ref_map[..., 1] * 100.0 / 255).astype(np.int8)
        self.ref = np.flip(self.ref, axis=1)

        self.static_map = np.zeros(self.ref.shape)
        self.static_map[:, :] = 10000
        self.static_map[self.ref == 100] = 1  # road
        self.static_map[(self.ref > 94) & (self.ref < 100)] = 50  # sidewalk
        self.static_map[self.ref == 70] = 100  # obstacle

    def get_costmap(self, agents):
        costmap = np.copy(self.static_map)
        for agent in agents:
            vertices = agent[1]
            if agent[0] == "car":
                x_min = min(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
                x_max = max(vertices[0][0], vertices[1][0], vertices[2][0], vertices[3][0])
                y_min = min(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
                y_max = max(vertices[0][1], vertices[1][1], vertices[2][1], vertices[3][1])
                for x in range(x_min, x_max):
                    for y in range(y_min, y_max):
                        costmap[x, y] = 100
            else:
                costmap[vertices[0], vertices[1]] = 10000
        return costmap


if __name__ == '__main__':
    grid = OccupancyGrid("Town01")
    cp = grid.get_costmap([["car", [[85, 1750], [90, 1750], [90, 1740], [85, 1740]]], ["pedestrian", [85, 1750]]])
    # cp[cp == 10000] = -56
    plt.imshow(cp, cmap="gray")
    plt.show()
