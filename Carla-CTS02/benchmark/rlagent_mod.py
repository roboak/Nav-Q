from PIL import Image

from benchmark.rlagent import RLAgent
import numpy as np

class RLAgent_mod(RLAgent):

    def __init__(self, world, carla_map, scenario):
        super().__init__(world, carla_map, scenario)

    def get_car_intention(self, obstacles, path, start):
        car_intention = super().get_car_intention(obstacles, path, start)
        return np.asarray(Image.fromarray(car_intention, 'RGB').crop(box=(150, 80, 250, 300))).copy()
