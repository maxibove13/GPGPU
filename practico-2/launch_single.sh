#!/bin/bash
#SBATCH --job-name=mitrabajo
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=00:01:00
#SBATCH --partition=normal
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=mi@correo
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

source /etc/profile.d/modules.sh

cd ~/mi_directorio

$1 $2
