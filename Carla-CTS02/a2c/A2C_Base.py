from abc import ABC
# from datetime import time
import time

from linetimer import CodeTimer
from torch.distributions import Categorical
from tqdm import trange
import cv2
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from a2c.classical_model.a2c_model import A2C
from config import Config
from utils.utils import plot_prob, l2_distance
import numpy as np
class A2CAbstract(ABC):
    '''
    Simulate an episode for 'num_steps' and save the value, log probability of action sampled, reward and entropy
    after every step of simulation

    :param num_steps: maximum number of steps in an episode
    :param model: RL model
    :param env: Environment where the agent is simulated
    :param device:
    :param latent_space_dim: Dimension of output of LSTM
    :param display: Boolean determining if the simulation should be displayed
    :param deter: Flag to set if the action should be detrmined stochastically or deterministically. Set this flag to True only at the time of evaluation
    :return: All the data saved during the episode
    '''
    def _simulate_trajectory(self, num_steps, model, env, device, latent_space_dim, display=False, deter=False):
        # Information used for training
        values = []
        log_probs = []
        rewards = []
        entropies = []

        # Addititional trajectory information
        speed_list = []
        trajectory = []
        actions_list = []
        reward = 0
        speed_action = 1
        velocity_x = 0
        velocity_y = 0
        nearmiss = False
        acccident = False
        observation = env.reset()
        begin_pos = env.world.player.get_location()
        total_episode_reward = 0
        step_num=0
        # Setup initial inputs for LSTM Cell
        cx = torch.zeros(1, latent_space_dim).type(torch.FloatTensor).to(device)
        hx = torch.zeros(1, latent_space_dim).type(torch.FloatTensor).to(device)
        steps_pbar = trange(int(num_steps), unit=" step", leave=True, position=0)
        for _ in steps_pbar:
            trajectory.append((env.world.player.get_location().x,
                               env.world.player.get_location().y))
            step_num +=1
            input_tensor = torch.from_numpy(observation).type(torch.FloatTensor).to(device)
            cat_tensor = torch.from_numpy(np.array([reward, velocity_x, velocity_y,
                                                    speed_action])).type(torch.FloatTensor).to(device)

            # with CodeTimer("Foward Pass"):
            logit, value, (hx, cx) = model(input_tensor, (hx, cx), cat_tensor)
            prob = F.softmax(logit, dim=-1)
            m = Categorical(prob)
            action = m.sample()
            # speed_action = action.item()
            if deter:
                speed_action = torch.argmax(logit, dim=1)[0].item()
            else:
                speed_action = action.item()
            actions_list.append(speed_action)

            if display:
                env.render()
                cv2.namedWindow(winname='CarIntention')
                cv2.imshow('CarIntention', observation)
                plot_prob(prob.squeeze(0).detach().numpy(), speed_action)
                cv2.waitKey(1)

            # Recording values for the next step
            # t = time.time()
            observation, reward, done, info = env.step(speed_action)
            # print("Time taken for simulation: {:.4f}sec".format((time.time() - t)))

            velocity = info['velocity']
            velocity_x = abs(velocity.x)
            velocity_y = abs(velocity.y)

            nearmiss_current = info['near miss']
            nearmiss = nearmiss_current or nearmiss
            acccident_current = info['accident']
            acccident = acccident_current or acccident

            # Used for logging
            total_episode_reward += reward

            # Logging value for loss calculation and backprop training
            log_prob = m.log_prob(action)
            entropy = -(F.log_softmax(logit, dim=-1) * prob).sum()
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            # vel converted from m/s to km /h for display
            vel = pow(velocity.x * velocity.x + velocity.y * velocity.y, 0.5) * 3.6

            speed_list.append(vel)
            steps_pbar.set_description("reward = {}, car_velocity = {}".format(reward, vel))
            if done or acccident:
                break

        time_to_goal = (step_num ) * Config.simulation_step

        # Calculate distance traveled
        end_pos = env.world.player.get_location()
        dist = l2_distance(begin_pos, end_pos)

        rewards.reverse()
        values.reverse()
        log_probs.reverse()
        entropies.reverse()

        trajectory_info = {}
        trajectory_info['trajectory'] = trajectory
        trajectory_info['actions'] = actions_list
        trajectory_info['impact_speed'] = speed_list
        trajectory_info['ttg'] = time_to_goal
        trajectory_info['total_episode_reward'] = total_episode_reward
        trajectory_info['goal'] = info['goal']
        trajectory_info['nearmiss'] = nearmiss
        trajectory_info['crash'] = info['accident']
        trajectory_info['ped_speed'] = info['ped_speed']
        trajectory_info['ped_dist'] = info['ped_distance']
        trajectory_info['scenario'] = info['scenario']
        trajectory_info['dist'] = dist
        
        training_info = {}
        training_info["rewards"] = rewards
        training_info["values"] = values
        training_info["log_probs"] = log_probs
        training_info["entropies"] = entropies
        return training_info, trajectory_info


    def close(self):
        '''
        Shut down carla server by deleting the environment
        :return:
        '''
        self.env.close()
