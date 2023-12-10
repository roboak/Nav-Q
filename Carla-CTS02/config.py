"""
Author: Dikshant Gupta
Time: 25.07.21 09:57
"""


class Config:
    PI = 3.14159

    simulation_step = 0.05  # 0.008
    # sensor_simulation_step = '0.5'
    synchronous = True
    segcam_fov = '90'
    segcam_image_x = '400'  # '1280'
    segcam_image_y = '400'  # '720'

    # # grid_size = 2  # grid size in meters
    # speed_limit = 50
    # max_steering_angle = 1.22173  # 70 degrees in radians
    # occupancy_grid_width = '1920'
    # occupancy_grid_height = '1080'

    # location_threshold = 1.0

    ped_speed_range = [0.6, 2.0]
    ped_distance_range = [0, 40]
    # car_speed_range = [6, 9]
    # scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    scenarios = ['01', '03', '04', '06', '07', '08']
    # scenarios = ['01']

    val_scenarios = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
    # val_scenarios = ['01'] #, '02', '03', '04', '05', '06', '07', '08']
    val_ped_speed_range = ([0.2, 0.5], [2.1, 2.8])
    val_ped_distance_range = [4.25, 49.25]
    val_car_speed_range = [6, 9]

    test_scenarios = ['01', '02', '03', '04', '05', '06', '07', '08']
    # test_scenarios = ['01', '02', '03']
    test_ped_speed_range = [0.25, 2.85]

    test_ped_distance_range = [4.75, 49.75]

    # Simulator Parameters
    host = 'localhost'
    port = 2000
    width = 1280
    height = 720
    display = False
    filter = 'vehicle.audi.tt'
    rolename = 'hero'
    gama = 1.7
    despot_port = 1245
    N_DISCRETE_ACTIONS = 3

    # reward function design
    max_speed = 50  # in kmph
    hit_penalty = 100
    near_miss_penalty = 10
    goal_reward = 200
    braking_penalty = 1
    over_speeding_penalty = 10

    # utils_parameters
    model_checkpointing_interval = 15
    max_checkpoints = 10

    # Target Entropy Scheduler Parameter
    exp_win_discount = 0.999
    avg_threshold = 0.01
    std_threshold = 0.05
    entropy_discount_factor = 0.9

    # A2C training parameters
    a2c_lr = 0.0005
    a2c_lr_initial = 1.0e-4
    a2c_lr_final = 5.0e-5
    a2c_gamma = 0.99
    # a2c_gae_lambda = 1.0
    a2c_entropy_coef = 0.0
    num_steps = 500
    train_episodes = 5000
    start_entropy_coeff = 0.0025
    end_entropy_coeff = 0.00001
    # a2c_eval_start_point = 1
    # a2c_eval_frequency = 1
    a2c_eval_start_point = 1000
    a2c_eval_frequency = 50
    a2c_eval_max_mean_ep_reward = float('-inf')