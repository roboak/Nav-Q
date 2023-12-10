from collections import OrderedDict
import os
import torch
import pickle as pkl
from a2c.A2C_Base import A2CAbstract
from a2c.classical_model.a2c_model import A2C
from config import Config
from benchmark.environment import GIDASBenchmark

class A2CEvaluator(A2CAbstract):
    def __init__(self, args):

        self.latent_space_dim = args.latent_space_dim

        # these variables are needed if the program terminates abruptly
        self.data_log = {}

        # Path to load model
        self.path = args.checkpoint
        if not os.path.exists(self.path):
            print("Path: {} does not exist".format(self.path))

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Setting up environment in eval mode
        self.env = GIDASBenchmark(port=args.port)
        self.env.eval()

        # Instantiating RL agent
        if (str(self.device) == "cpu"):
            orig_model_dict = torch.load(self.path, map_location=torch.device("cpu"))
        else:
            orig_model_dict = torch.load(self.path)
        pruned_state_dict = {}

        # We create an object of classical model for evaluating both classical and quantum models because, the critic
        # is not used at the time of training
        self.model = A2C(hidden_dim=args.latent_space_dim, num_actions=3, model_eval=True).to(
                self.device)

        # Remove the keys from weight dictionary that belong to the weights of the critic
        for key, value in orig_model_dict.items():
            if 'value' not in key:
                pruned_state_dict[key] = value
        model_dict = OrderedDict(pruned_state_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()

        self.data_log[0] = str(vars(Config)) + str(vars(args))

    def eval(self, args, filename, current_episode=0):
        try:
            # Simulation loop
            max_episodes = len(self.env.episodes)
            print("Total eval episodes: {}".format(max_episodes))
            # data_log = {}

            while current_episode < max_episodes:
                training_info, trajectory_info = self._simulate_trajectory(Config.num_steps, self.model, self.env, self.device,
                                                self.latent_space_dim, display=args.display, deter=args.deter)

                self.data_log[current_episode + 1] = trajectory_info

                print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(
                    current_episode + 1, trajectory_info['scenario'], trajectory_info['ped_speed'], trajectory_info['ped_dist']))
                print('Goal reached: {}, Time to goal: {:.4f}s, Accident: {}, Nearmiss: {}, Reward: {:.4f}'.format(
                    trajectory_info['goal'], trajectory_info['ttg'], trajectory_info['crash'], trajectory_info['nearmiss'], trajectory_info['total_episode_reward']))
                current_episode += 1

            self.close()
            with open(filename, "wb") as write_file:
                pkl.dump(self.data_log, write_file)
            print("Log file written here: {}".format(filename))
            print('-' * 60)
        except BaseException as e:
            print(e)
            self.close()
            with open(filename, "wb") as write_file:
                pkl.dump(self.data_log, write_file)
            print("Log file written here: {}".format(filename))
            print('-' * 60)

