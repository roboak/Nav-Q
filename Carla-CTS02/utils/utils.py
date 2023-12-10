"""
Author: Akash Sinha
Time: 23.03.21 14:29
"""
import re
import carla
import math
import torch
import cv2
import numpy as np
from config import Config
import os
import subprocess
from datetime import datetime
from pathlib import Path

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly.

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)

def plot_prob(prob, action):
    image = np.zeros((400, 400, 3), np.uint8)
    color = (0, 255, 0)
    for i in range(len(prob)):
        x = (i + 1) * 50
        y = 0
        w = 50
        h = int(prob[i] * 400)
        cv2.rectangle(image, (x, h), (x + w, 0), color, -1)
    cv2.putText(image, str(action), (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    cv2.imshow('Action Probability', image)

def l2_distance(pos1, pos2):
    '''

    :param pos1: (x,y) coordinates of position 1
    :param pos2: (x,y) coordinates of position 2
    :return: Euclidean distance between two points
    '''
    direction = pos1 - pos2
    direction_norm = math.sqrt(direction.x ** 2 + direction.y ** 2)
    return direction_norm

def clear_checkpoints(model_dir: str):
    '''
    Method to remove old model checkpoints to save memory
    :param model_dir: Directory where the checkpoints are saved
    :return: None
    '''
    saved_models = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
    while len(saved_models) > Config.max_checkpoints:
        x = saved_models.pop(0)
        os.remove(x)

def run_server(local: bool,  port):
    '''
    :param local: If local flag is set, CARLA server server on local computer with display using executable file's path.
    Otherwise, CARLA server is run on headless VMs.
    :param port: Port for CARLA server
    :return: None
    '''
    port = "-carla-port={}".format(port)
    if local:
        print("executing locally")
        subprocess.run("cd D:/CARLA_0.9.13/WindowsNoEditor && CarlaUE4.exe " + port, shell=True)
    else:
        print("executing on slurm cluster")
        subprocess.run(['cd /netscratch/sinha/carla && unset SDL_VIDEODRIVER && ./CarlaUE4.sh -vulkan -RenderOffscreen -nosound ' + port], shell=True)





def _get_dir_name(args, quantum=True):
    if quantum:
        return '_q_' + str(args.n_qubits) + '_l_' + str(args.n_layers) + '_ls_' + str(args.latent_space_dim) + '_seed_' + str(args.seed)
    else:
        return '_ls_' + str(args.latent_space_dim) + '_seed_' + str(args.seed)

def get_run_dir(args):

    # Based on parameters, define the name of the rl_model. This will be later used in creating run directories
    rl_model = ""
    if args.a2c:
        rl_model = "a2c"
    if args.ppo:
        rl_model = "ppo"
    # if batching is used - This is used when batches of experiments with different quantum hyper parameters have to be run.
    if args.batch_name:
        if args.checkpoint:
            # The checkpoint directory will mark the root directory where all the batch experiments will be stored.
            # NOT the PATH of any particular model
            # Example: ../batch_name
            sub_dirs = [x[0] for x in os.walk(args.checkpoint)]
            # run_dir is one of the sub directories of the batch_dir
            # run_dir is chosen based on command line params - n_qubits, n_layers, latent_Space_dim
            # For example: # ../batch_name/date_time_q_#_l_#_ls_#_seed_#
            dir_found = False
            for x in sub_dirs:
                dir_name = ""
                if args.quantum:
                    dir_name = 'q_{}_l_{}_ls_{}_seed_{}'.format(args.n_qubits, args.n_layers, args.latent_space_dim, args.seed)
                else:
                    dir_name = '_ls_{}_seed_{}'.format(args.latent_space_dim, args.seed)

                if dir_name in x:
                    dir_found = True
                    run_dir = x
                    model_dir = os.path.join(run_dir, "model")
                    # args.checkpoint should finally point to the latest model
                    model_paths = sorted(Path(model_dir).iterdir(), key=os.path.getmtime)
                    if len(model_paths)==0:
                        # directory was created but when there are no saved checkpoints
                        args.checkpoint = None
                    else:
                        # args.checkpoint should finally point to the latest model
                        args.checkpoint = model_paths[-1].as_posix()
                    break
            if not dir_found:
                # When the experiment failed and the directory could not be created; we create a new run_dir
                run_dir = os.path.join(args.checkpoint, datetime.now().strftime("%m%d%Y_%H%M%S")+ _get_dir_name(args, quantum=args.quantum))
                args.checkpoint = None

        else:
            run_dir = "_out/{}/{}/{}".format(rl_model, args.batch_name, datetime.now().strftime("%m%d%Y_%H%M%S")+\
                                           '_q_' + str(args.n_qubits) + '_l_' + str(args.n_layers) + '_ls_' + str(
                args.latent_space_dim))
    else:
        if args.checkpoint:
            # The checkpoint directory should point to the model..looks like this: .../run_dir/model/model.pth
            run_dir = os.sep.join(args.checkpoint.split(os.sep)[:-2])
        else:
            run_dir = "_out/{}/{}_{}".format(rl_model, args.run_id, datetime.now().strftime("%m%d%Y_%H%M%S"))

    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir
