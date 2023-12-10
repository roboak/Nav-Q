import argparse
import time
import os
from multiprocessing import Process
from datetime import datetime
from a2c.A2C_Evalulator import A2CEvaluator
from config import Config
from utils.utils import run_server


def main():
    # print(__doc__)
    parser = argparse.ArgumentParser(
        description='CARLA A2C Control Client')

    parser.add_argument(
        '-ckp', '--checkpoint',
        default='',
        type=str,
        help='load the model from this checkpoint')

    # Port on which CARLA server will run
    parser.add_argument('-p', '--port', metavar='P', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    # Use this argument when running on local computer and when the CARLA server should be started my the main program.
    parser.add_argument('--local', action='store_true',
                        help="Use this argument when running on local computer and when the CARLA server should be started my the main program.")
    # Use this argument when running on local computer and the CARLA server is already running
    parser.add_argument('--debug', action='store_true',
                        help="Use this argument when running on local computer and the CARLA server is already running")
    parser.add_argument('--latent_space_dim', type=int, default=32)
    # Used as an argument to display. This argument should never be used when training the model on headless server
    parser.add_argument('--display', action='store_true',
                        help="Used as an argument to display. This argument should never be used when training the model on headless server")

    # Parameters to decide the model that should be evaluated
    parser.add_argument('--a2c', action='store_true', help="Use this flag for selecting A2C model")
    parser.add_argument('--ppo', action='store_true', help="Use this flag for selecting PPO model")

    # Used to choose if the action should be determined deterministically at test time
    parser.add_argument('--deter', action='store_true',
                        help='Used to choose if the action should be determined deterministically at test time')
    # evaluation would start form this point
    parser.add_argument('--ep', type=int, default=0)

    # Read command line arguments
    args = parser.parse_args()
    Config.port = args.port
    print("port: ", Config.port)

    # Start CARLA server
    if (not args.debug):
        p = Process(target=run_server, args=(args.local, args.port,))
        p.start()
        time.sleep(100)  # wait for the server to start

    # Name of the pickel file which will store the evaluation data
    filename = "{}.pkl".format(datetime.now().strftime("%m%d%Y_%H%M%S"))
    pkl_path = os.sep.join(args.checkpoint.split(os.sep)[:-2])
    filename = os.path.join(pkl_path, filename)
    print(filename)
    Evaluator = None
    if args.a2c:
        Evaluator = A2CEvaluator(args)
    else:
        raise Exception("RL model not implemented")

    Evaluator.eval(args, filename, current_episode=args.ep)


if __name__ == '__main__':
    main()




