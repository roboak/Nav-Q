
import subprocess
import time
from multiprocessing import Process
def run_server(local: bool, mode = str):
    # train environment
    if mode == "train":
        port = "-carla-port={}".format(2000)
    else:
        port = "-carla-port={}".format(2100)
    if local:
        print("executing locally")
        subprocess.run("cd D:/CARLA_0.9.13/WindowsNoEditor && CarlaUE4.exe " + port, shell=True)
    else:
        print("executing on slurm cluster")
        subprocess.run(['cd /netscratch/sinha/carla && unset SDL_VIDEODRIVER && ./CarlaUE4.sh -vulkan -RenderOffscreen -nosound ' + port], shell=True)

if __name__ == '__main__':
    p = Process(target=run_server, args=(True, "train", ))
    p.start()
    time.sleep(12)
    # p = Process(target=run_server, args=(True, "test", ))
    # p.start()
    # time.sleep(12)
    #
