#!/bin/bash
#SBATCH --job-name Q_A2C_ls_32_q_4_l_1_multi_runs
#SBATCH --output=%j.log
#SBATCH --partition=RTX3090
#SBATCH --time=03-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=70GB
#SBATCH --array=0-4%5
#SBATCH --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
# Define the parameter lists
#TODO: Line 2 must be updated
#TODO: Line 9 must be updated
#TODO: Lines 15, 16, 17, 18 must be updated
latent_spaces=(32)
n_layers=(1)
n_qubits=(4)
seeds=(51 52 53 67 95)
#seeds=(51 52 67)
#port numbers
lower=2400
upper=2500

job_index=$SLURM_ARRAY_TASK_ID
latent_space_index=$(((job_index/$((${#seeds[@]}*${#n_layers[@]}*${#n_qubits[@]})))%${#latent_spaces[@]}))
layer_index=$(((job_index/(${#seeds[@]}*${#n_qubits[@]}))%${#n_layers[@]}))
#qubit_index=$((job_index % ${#n_qubits[@]}))
qubit_index=$(((job_index/${#seeds[@]}) % ${#n_qubits[@]}))
seed_index=$((job_index % ${#seeds[@]}))

latent_space_dim=${latent_spaces[$latent_space_index]}
layers=${n_layers[$layer_index]}
q=${n_qubits[$qubit_index]}
seed=${seeds[$seed_index]}

username="$USER"
IMAGE=/netscratch/$USER/vanilla.sqsh
WORKDIR="/netscratch/sinha/Q-DRL-for-CFN"
# TODO: The following line (40) must be updated everytime.
batch_name="Q_A2C_ls_32_q_4_l_1_multi_runs"
batch_checkpoint="_out/a2c/${batch_name}"
#srun \
#  --container-image=$IMAGE \
#  --container-workdir=$WORKDIR\
#  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/sinha:/home/sinha \
#  --no-container-remap-root \
#  --job-name Q_A2C_layers_${layers}_ls_${latent_space_dim}_q_${n_qubits} \
#  /home/sinha/miniconda3/envs/qns/bin/python3 train_a2c.py \
#  --latent_space_dim="$latent_space_dim" \
#  --n_qubits="$q" \
#  --n_layers="$layers" \
#  --quantum \
#  --port=$(shuf -i $lower-$upper -n 1) \
#  --batch_name=${batch_name} \
#  --ansatz 3

#If quantum model has to be used checkpoints have to be loaded, then use the following command
srun \
  --container-image=$IMAGE \
  --container-workdir=$WORKDIR\
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/sinha:/home/sinha \
  --no-container-remap-root \
  --job-name Q_A2C_layers_${layers}_ls_${latent_space_dim}_q_${n_qubits} \
   /netscratch/sinha/c_qns/bin/python3 train_a2c.py \
  --latent_space_dim="$latent_space_dim" \
  --n_qubits="$q" \
  --n_layers="$layers" \
  --quantum \
  --ansatz 1 \
  --port=$(shuf -i $lower-$upper -n 1) \
  --batch_name=${batch_name} \
  --checkpoint=${batch_checkpoint} \
  --seed=${seed} \
#  --a2c

#If a batch of classical models have to be trained with different seeds
#srun \
#  --container-image=$IMAGE \
#  --container-workdir=$WORKDIR\
#  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,/home/sinha:/home/sinha \
#  --no-container-remap-root \
#  --job-name Q_A2C_layers_${layers}_ls_${latent_space_dim}_q_${n_qubits} \
#  /home/sinha/miniconda3/envs/qns/bin/python3 train_a2c.py \
#  --latent_space_dim="$latent_space_dim" \
#  --port=$(shuf -i $lower-$upper -n 1) \
#  --batch_name=${batch_name} \
#  --checkpoint=${batch_checkpoint} \
#  --seed=${seed}