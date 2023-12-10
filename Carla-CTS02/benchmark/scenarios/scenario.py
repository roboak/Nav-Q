"""
Author: Dikshant Gupta
Time: 21.08.21 10:08
"""

import random
import carla


class Scenario:
    def __init__(self, world):
        self.world = world

    def scenario12(self):
        start = (2, 250, -90)
        end = (2, 150, -90)
        # end = (2, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 100
        walker_spawn_point.location.y = 300
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = -1.5
        car_spawn_point.location.y = 150
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        parked_car_spawn_point = carla.Transform()
        parked_car_spawn_point.location.x = 2
        parked_car_spawn_point.location.y = 200
        parked_car_spawn_point.location.z = 0.01
        parked_car_spawn_point.rotation.yaw = -90
        parked_car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        parked_car = [random.choice(parked_car_bp), parked_car_spawn_point]
        obstacles.append(parked_car)

        return 12, obstacles, end, start

    def scenario11(self):
        start = (-2, 5, 90)
        # start = (10, -2, -180)
        end = (-2, 100, 90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 100
        walker_spawn_point.location.y = 300
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = -2
        car_spawn_point.location.y = 60
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 270
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        parked_car_spawn_point = carla.Transform()
        parked_car_spawn_point.location.x = 2
        parked_car_spawn_point.location.y = 10
        parked_car_spawn_point.location.z = 0.01
        parked_car_spawn_point.rotation.yaw = 270
        parked_car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        parked_car = [random.choice(parked_car_bp), parked_car_spawn_point]
        obstacles.append(parked_car)

        return 11, obstacles, end, start

    def scenario10(self):
        start = (2, 270, -90)
        end = (2, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 4.5  # 2 For testing purposes
        walker_spawn_point.location.y = 230
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = -1.5
        car_spawn_point.location.y = 130
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        return 10, obstacles, end, start

    def scenario09(self):
        start = (92, 160, -90)
        end = (60, -2, -180)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 75
        walker_spawn_point.location.y = -4.5
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 90.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 9, obstacles, end, start

    def scenario08(self):
        start = (-2.0, 270, -90)
        end = (-2.0, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 5.2
        walker_spawn_point.location.y = 200
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = 4.8
        car_spawn_point.location.y = 206
        car_spawn_point.location.z = 0.5
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        return 7, obstacles, end, start

    def scenario07(self):
        start = (-2.0, 270, -90)
        end = (-2.0, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 4.0
        walker_spawn_point.location.y = 200
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = 2
        car_spawn_point.location.y = 206
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        return 7, obstacles, end, start

    def scenario06(self):
        start = (92, 2, 90)
        end = (92, 120, 90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 96
        walker_spawn_point.location.y = 72
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 6, obstacles, end, start

    def scenario05(self):
        start = (88.0, 260, -90)
        end = (88.0, 140, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 94
        walker_spawn_point.location.y = 190
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 5, obstacles, end, start

    def scenario04(self):
        start = (-2.0, 270, -90)
        end = (-2.0, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 4.0
        walker_spawn_point.location.y = 200
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 4, obstacles, end, start

    def scenario03(self):
        start = (2, 270, -90)
        end = (2, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = -4.0
        walker_spawn_point.location.y = 203
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        car_spawn_point = carla.Transform()
        car_spawn_point.location.x = -2
        car_spawn_point.location.y = 206
        car_spawn_point.location.z = 0.01
        car_spawn_point.rotation.yaw = 90
        car_bp = self.world.get_blueprint_library().filter("vehicle.audi.tt")
        car = [random.choice(car_bp), car_spawn_point]
        obstacles.append(car)

        return 3, obstacles, end, start

    def scenario02(self):
        start = (92, 160, -90)
        end = (92, 2, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = 86.5
        walker_spawn_point.location.y = 90
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 2, obstacles, end, start

    def scenario01(self):
        start = (2, 270, -90)
        end = (2, 150, -90)
        obstacles = []

        walker_bp = self.world.get_blueprint_library().filter("walker.pedestrian.0001")
        walker_spawn_point = carla.Transform()
        walker_spawn_point.location.x = -4.0
        walker_spawn_point.location.y = 200
        walker_spawn_point.location.z += 1.0
        walker_spawn_point.rotation.yaw = 270.0
        walker = [random.choice(walker_bp), walker_spawn_point]
        obstacles.append(walker)

        return 1, obstacles, end, start