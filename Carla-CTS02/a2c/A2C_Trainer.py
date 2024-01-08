from multiprocessing import Process
import yaml
from a2c.A2C_Base import A2CAbstract
from pathlib import Path
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from a2c.classical_model.a2c_model import A2C
from config import Config
from benchmark.environment import GIDASBenchmark
from a2c.quantum_model.q_a2c_model import Q_A2C
from utils.utils import get_run_dir, run_server


class A2CTrainer(A2CAbstract):

    def __init__(self, args):

        self.args = args
        ##############################################################
        # Start CARLA server
        if (not args.debug):
            p = Process(target=run_server, args=(args.local, args.port,))
            p.start()
            time.sleep(100)  # wait for the server to start

        ##############################################################
        #Setup Environment
        self.env = GIDASBenchmark(port=args.port)

        ##############################################################
        # Choose between GPU or CPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device, " being used")

        ##############################################################
        #Initialise RL model
        if args.quantum:
            # self.device = torch.device("cpu")
            self.model = Q_A2C(n_layers=args.n_layers, hidden_dim=args.latent_space_dim,
                               n_qubits=args.n_qubits, ansatz=args.ansatz, depolarising_error=args.dep_err, gate_control_noise=args.gate_noise).to(self.device)
        else:
            self.model = A2C(hidden_dim=args.latent_space_dim, num_actions=3).to(self.device)

        ##############################################################
        # If training has to be resumed from a saved checkpoint
        self.current_episode = 0
        load_path = args.checkpoint
        print("checkpoint:", load_path)
        if load_path:
            # Checkpoint name can be saved as following "_xx_xx_xx_episode_num.pth "
            self.current_episode = int(load_path.split(os.sep)[-1].split('_')[-1].split('.')[0])
            if (str(self.device) == "cpu"):
                self.model.load_state_dict(torch.load(load_path, map_location=torch.device("cpu")))
            else:
                self.model.load_state_dict(torch.load(load_path))

        ##############################################################
        # Setting up logging directories within run_dir
        run_dir = get_run_dir(args)
        # Summary_Dir will contain tensorboard logs
        summary_dir = os.path.join(run_dir, "summary")
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)

        # Setting up tensorboard
        self.writer = SummaryWriter(log_dir=summary_dir)

        # Path to save checkpoints
        self.model_path = os.path.join(run_dir, "model")
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        ##############################################################
        # Save the args and config parameters for future references

        config_dict = {key: value for key, value in vars(Config).items() if
                       not callable(value) and not key.startswith("__")}
        args.config_dict = config_dict
        n_params = {}
        n_params["shared_network"] = sum(p.numel() for p in self.model.shared_network.parameters())
        n_params["actor_network"] = sum(p.numel() for p in self.model.action_policy.parameters())
        n_params["critic_network"] = sum(p.numel() for p in self.model.value_network.parameters())

        args.params = n_params
        file = open(os.path.join(run_dir, "experiment.yaml"), "w")
        yaml.dump(args, file)
        file.close()

    def _clear_checkpoints(self, model_dir):
        saved_models = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
        while len(saved_models) > Config.max_checkpoints:
            x = saved_models.pop(0)
            os.remove(x)

    def _calc_loss(self, rewards, values, log_probs, entropies, device):
        R = 0
        returns = []
        for r in rewards:
            R = Config.a2c_gamma * R + r
            if R < -1:
                R = -1
            returns.append(R)
        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item()

        # Returns vector is normalised to have 0 mean and 1 std
        returns = (returns - returns.mean()) / (returns.std() + eps)
        returns = returns.type(torch.FloatTensor).to(device)

        policy_losses = []
        value_losses = []

        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]]).to(device)))

        loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean() - \
               Config.a2c_entropy_coef * torch.stack(entropies).mean()
        return loss, sum(policy_losses) / len(policy_losses), sum(value_losses) / len(value_losses)

    def _log_metrics(self, writer, current_episode, policy_loss, value_loss, total_episode_reward, dist, entropy, goal,
                     acccident, nearmiss, scenario, ped_speed, ped_distance):
        '''
        Logging the required metrics in tensorboard
        :param policy_loss: Actor Loss
        :param value_loss: Critic Loss
        :param total_episode_reward: Sum of rewards attained in the previous episode
        :param dist: Distance travelled by the car in the previous episode
        :param entropy: Average entropy of the entire episode
        :param acccident: Boolean to store if accident happened in the last episode
        :param nearmiss: Boolean to store if there were near misses in the last episode
        :return: None
        '''
        writer.add_scalar(tag="Loss/Policy", scalar_value=policy_loss, global_step=current_episode)
        writer.add_scalar(tag="Loss/Critic", scalar_value=value_loss, global_step=current_episode)
        writer.add_scalar(tag="Return/Reward", scalar_value=total_episode_reward, global_step=current_episode)
        writer.add_scalar(tag="Stats/Distance", scalar_value=dist, global_step=current_episode)
        writer.add_scalar(tag="Stats/Entropy", scalar_value=entropy, global_step=current_episode)

        print("Episode: {}, Scenario: {}, Pedestrian Speed: {:.2f}m/s, Ped_distance: {:.2f}m".format(current_episode,
                                                                                                     scenario,
                                                                                                     ped_speed,
                                                                                                     ped_distance))
        print('Goal reached: {}, Accident: {}, Nearmiss: {}'.format(goal, acccident, nearmiss))
        print("Policy Loss: {:.4f}, Value Loss: {:.4f}, Entropy: {:.4f}, Reward: {:.4f}, Dist_car: {}".format(
            policy_loss.item(), value_loss.item(),
            entropy, total_episode_reward, dist))

    def run_training(self):
        t0 = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.a2c_lr)
        last_save = 0
        ##############################################################
        # Simulation loop
        max_episodes = Config.train_episodes
        print("Total training episodes: {}".format(max_episodes))

        while self.current_episode < max_episodes:
            training_info, trajectory_info = self._simulate_trajectory(num_steps=Config.num_steps, model=self.model, env=self.env, device=self.device, latent_space_dim=32)
            rewards = training_info["rewards"]
            values = training_info["values"]
            log_probs = training_info["log_probs"]
            entropies = training_info["entropies"]
            acccident = trajectory_info["crash"]
            nearmiss = trajectory_info["nearmiss"]
            dist = trajectory_info["dist"]
            total_episode_reward = trajectory_info["total_episode_reward"]
            loss, policy_loss, value_loss = self._calc_loss(rewards, values, log_probs, entropies, self.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            self._log_metrics(self.writer, self.current_episode, policy_loss, value_loss,
                         total_episode_reward, dist, torch.stack(entropies).mean(), trajectory_info["goal"],
                         acccident, nearmiss, trajectory_info['scenario'], trajectory_info['ped_speed'],
                         trajectory_info['ped_dist'])

            # Model checkpointing
            if self.current_episode - last_save > Config.model_checkpointing_interval and total_episode_reward > 0:
                last_save = self.current_episode
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_path,
                                        "a2c_{}_{}.pth".format(total_episode_reward, self.current_episode)))
                self._clear_checkpoints(model_dir=self.model_path)

            self.current_episode += 1

        self.env.close()
        print("Training time: {:.4f}hrs".format((time.time() - t0) / 3600))
        # file.write("Training time: {:.4f}hrs\n".format((time.time() - t0) / 3600))
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "a2c_entropy_{}.pth".format(self.current_episode)))

