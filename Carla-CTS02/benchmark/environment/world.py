"""
Author: Dikshant Gupta
Time: 23.03.21 14:27
"""
import sys
import random

from benchmark.environment.utils import find_weather_presets
from benchmark.environment.sensors import *


class World(object):
    def __init__(self, carla_world, hud, scenario, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.scenario = None
        self.player = None
        self.walker = None
        self.incoming_car = None
        self.parked_cars = None
        self.player_max_speed = None
        self.player_max_speed_fast = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.semseg_sensor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gama
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        self.car_blueprint = self.get_car_blueprint()
        self.ped_speed = None
        self.ped_distance = None
        self.restart(scenario)
        self.world.on_tick(hud.on_world_tick)
        for _ in range(2):
            self.next_weather()

    def get_car_blueprint(self):
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = blueprint.get_attribute('color').recommended_values[1]
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        return blueprint

    def get_random_blueprint(self):
        vehicles = ["vehicle.audi.a2", "vehicle.audi.tt", "vehicle.chevrolet.impala", "vehicle.audi.etron"]
        vehicle_type = random.choice(vehicles)
        blueprint = random.choice(self.world.get_blueprint_library().filter(vehicle_type))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        return blueprint

    def restart(self, scenario, ped_speed=1, ped_distance=30):
        self.scenario = scenario
        self.ped_speed = ped_speed
        self.ped_distance = ped_distance

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        semseg_index = self.semseg_sensor.index if self.semseg_sensor is not None else 5
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 3
        semseg_pos_index = self.semseg_sensor.transform_index if self.semseg_sensor is not None else 3

        # Spawn the player.
        start = self.scenario[3]
        spawn_point = carla.Transform()
        spawn_point.location.x = start[0]
        spawn_point.location.y = start[1]
        spawn_point.location.z = 0.01
        spawn_point.rotation.yaw = start[2]


        if self.player is not None:
            self.destroy()
            self.player = self.world.try_spawn_actor(self.car_blueprint, spawn_point)
            # TODO:spawning the car with a certain initial velocity
            self.modify_vehicle_physics(self.player)
            # self.player.set_target_velocity(carla.Vector3D(0, random.randint(2, 6), 0))
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            self.player = self.world.try_spawn_actor(self.car_blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        # Set up other agents
        scenario_type = self.scenario[0]
        obstacles = self.scenario[1]
        if scenario_type in [1, 2, 4, 5]:
            # Single pedestrian scenarios
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, self.ped_speed, 0), 1))
        if scenario_type == 6:
            # Single pedestrian scenarios
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, -self.ped_speed, 0), 1))
        elif scenario_type in [3, 7, 8]:
            # Single pedestrian scenarios with parked car
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
        elif scenario_type == 10:
            # Single pedestrian with incoming car
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, -self.ped_speed, 0), 1))
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
        elif scenario_type == 9:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
        elif scenario_type == 11:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.player.set_target_velocity(carla.Vector3D(0, 20 * 0.2778, 0))
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
            self.parked_cars = []
            car_spawn_point = obstacles[2][1]
            car_spawn_point.location.y -= 7
            for _ in range(12):
                car_spawn_point.location.y += 7
                parked_car = None
                while parked_car is None:
                    parked_car = self.world.try_spawn_actor(self.get_random_blueprint(), car_spawn_point)
                self.parked_cars.append(parked_car)
        elif scenario_type == 12:
            self.walker = self.world.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.incoming_car = self.world.try_spawn_actor(obstacles[1][0], obstacles[1][1])
            self.parked_cars = []
            parked_car = None
            while parked_car is None:
                parked_car = self.world.try_spawn_actor(obstacles[2][0], obstacles[2][1])
            self.parked_cars.append(parked_car)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.semseg_sensor = CameraManager(self.player, self.hud, self._gamma)
        self.semseg_sensor.transform_index = semseg_pos_index
        self.semseg_sensor.set_sensor(semseg_index, notify=False)

    def tick(self, clock):
        self.hud.tick(self, clock)
        dist_walker = abs(self.player.get_location().y - self.walker.get_location().y)
        car_velocity = self.player.get_velocity()
        car_speed = np.sqrt(car_velocity.x ** 2 + car_velocity.y ** 2)
        if dist_walker < self.ped_distance:  # and car_speed > 0:
            if self.scenario[0] in [1, 2, 3]:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(self.ped_speed, 0, 0), 1))
                if self.scenario[0] in [1, 3] and self.walker.get_location().x > 4.5:
                    self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
                if self.scenario[0] == 2 and self.walker.get_location().x > 95.0:
                    self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
            elif self.scenario[0] in [4, 5, 7, 8, 6]:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(-self.ped_speed, 0, 0), 1))
                if self.walker.get_location().x < -4.5 and self.scenario[0] in [4, 7, 8]:
                    self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
                if self.scenario[0] in [5, 6] and self.walker.get_location().x < 85.0:
                    self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
            elif self.scenario[0] == 10:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(-self.ped_speed, 0, 0), 1))
            elif self.scenario[0] == 9:
                self.walker.apply_control(carla.WalkerControl(carla.Vector3D(0, self.ped_speed, 0), 1))
        if self.scenario[0] == 10:
            flag = (0 < (self.walker.get_location().y - self.incoming_car.get_location().y) < 5) and \
                   (self.walker.get_location().x > -4.4)
            if self.incoming_car.get_location().y > 250 or flag:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, 9, 0))  # Set target velocity for experiment
        if self.scenario[0] == 11:
            # self.incoming_car.set_target_velocity(carla.Vector3D(0, -20 * 0.2778, 0))
            if self.incoming_car.get_location().y - self.player.get_location().y < 10:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                self.incoming_car.set_target_velocity(carla.Vector3D(0, -self.ped_speed * 0.2778, 0))
        if self.scenario[0] == 12:
            # pass
            self.incoming_car.set_target_velocity(carla.Vector3D(0, self.ped_speed * 0.2778, 0))

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.world.set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def render(self, display):
        self.camera_manager.render(display)
        self.semseg_sensor.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

        self.semseg_sensor.sensor.destroy()
        self.semseg_sensor.sensor = None
        self.semseg_sensor.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.semseg_sensor.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.walker is not None:
            self.walker.destroy()
        # if self.incoming_car is not None and self.scenario[0] in [10, 3, 7, 8]:
        if self.incoming_car is not None and self.incoming_car.is_alive:
            self.incoming_car.destroy()
        if self.scenario[0] in [11, 12]:
            if self.parked_cars is not None:
                for car in self.parked_cars:
                    car.destroy()
