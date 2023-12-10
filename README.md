This repository includes the implementation for Nav-Q [[Sinha et al., 2023]][Nav-Q]

### Environment Setup
1. Install CARLA by following the instructions mentioned here([https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation][CARLA Installation]). 
This repository is compatible with CARLA 0.9.13
2. Create a python virtual environment and install all the required dependencies using the command `pip install -r requirements.txt`
3. Set CARLA-CTS02 as the working directory.

### Steps to Run

**Debug Mode**
This mode should be used for development environments. 
1. In this mode, CARLA server is started independently as a separate
process using the Carla-CTS02/utils/test_carla_server.py file. Make sure, that the path of the CARLA executable is updated.
2. Start training Nav-Q model using the following command
`python train.py --a2c --port 2000 --debug --latent_space_dim 32 --seed 54 --quantum --n_qubits 4 --n_layers 1` 
Note that this command is used to train Nav-Q with 4 qubits and 1 layer.
3. Train NavA2C using the following command
`python train.py --a2c --port 2000 --debug --latent_space_dim 32 --seed 54`
4. Evaluate Nav-Q/NavA2C using the following command
`python eval_model.py --a2c --port 2000 --debug --latent_space_dim 32 --checkpoint <path of the model> --deter `
Note that deter flag is used to set that the model determines actions deterministically at test time. 

### Notes
1. If the repo has to be executed on a cluster, then CARLA server has to be started at the time of training from within the program. In such a case, do not use 'debug' flag while running
the training and evaluation. 
2. Make sure to update the path of CARLA executable in "run_server" method in Carla-CTS02/utils/utils.py  




[CARLA Installation]: https://carla.readthedocs.io/en/latest/start_quickstart/#carla-installation

[Nav-Q]: https://arxiv.org/abs/2311.12875