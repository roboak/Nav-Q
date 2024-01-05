import sys
import pygame
import argparse
from a2c.A2C_Trainer import A2CTrainer
from config import Config


if __name__ == '__main__':

    sys.setrecursionlimit(5000)
    parser = argparse.ArgumentParser(
        prog="Nav-Q",
        description='Train Nav-Q algorithm using CARLA simulator')
    parser.add_argument('--seed', type=int, default=101)

    # Specify custom run_ids for each run
    parser.add_argument('--run_id', type=str, default='a2c', help="Specify custom run_ids for each run")
    '''Loading checkpoints. For individual runs, provide the path of the model. For restarting batch jobs from
    respective checkpoints,  provide the path of the batch folder
    '''
    parser.add_argument( '-ckp', '--checkpoint', default='',type=str,
        help='load the model from this checkpoint')

    # Port on which CARLA server will run
    parser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    # Use this argument when running on local computer and when the CARLA server should be started my the main program.
    parser.add_argument('--local', action='store_true',
                        help="Use this argument when running on local computer and when the CARLA server should be started my the main program.")
    # Use this argument when running on local computer and the CARLA server is already running
    parser.add_argument('--debug', action='store_true',
                        help="Use this argument when running on local computer and the CARLA server is already running")

    # Used as an argument to display. This argument should never be used when training the model on headless server
    parser.add_argument('--display', action='store_true',
                        help="Used as an argument to display. This argument should never be used when training the model on headless server")


    # Parameters to define the quantum parameters
    parser.add_argument('--quantum', action='store_true')
    parser.add_argument('--ansatz', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--latent_space_dim', type=int, default=12)
    parser.add_argument('--dep_err', action='store_true', help="Flag to enable depolarising error")
    parser.add_argument('--gate_noise', action='store_true', help="Flag to enable gate noise")

    # Flag to be set when running a2c RL algorithms
    parser.add_argument('--a2c', action='store_true', help="Flag to be set when running a2c RL algorithms")
    parser.add_argument('--ppo', action='store_true', help="Flag to be set when running a2c RL algorithms")

    # To be passed as a parameter when multiple slurm jobs are run using SBATCH command
    parser.add_argument('--batch_name', type=str, default = '')

    # Read command line arguments
    args = parser.parse_args()
    Config.port = args.port
    try:
        trainer = A2CTrainer(args)
        trainer.run_training()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        pygame.quit()


