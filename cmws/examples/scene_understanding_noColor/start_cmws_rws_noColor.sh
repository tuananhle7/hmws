#!/bin/bash

#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu
#SBATCH --mem=100G
#SBATCH -t 24:00:00
#SBATCH --mail-user=katiemc@mit.edu
#SBATCH --mail-type=ALL
#SBATCH -p tenenbaum
#SBATCH --output=/om/user/katiemc/continuous_mws/outputs/sweep_%A_%a.out
#SBATCH --error=/om/user/katiemc/continuous_mws/outputs/sweep_%A_%a.err

module load openmind/anaconda/3-2019.10;module load openmind/gcc/7.5.0; module load openmind/cuda/10.2; module load openmind/cmake/3.12.0; source activate cmws

expName="cmws_vs_rws_noColor"

python3 sweep_colorless.py &

python3 plot.py --experiment-name=$expName --repeat &

wait


