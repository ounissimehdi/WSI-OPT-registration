#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/REG-%A_%a.txt
#SBATCH --error=./bash-log/REG-%A_%a.txt
#SBATCH --job-name=S2:reg-LARGE
#SBATCH --array=0-218

python reg_large.py -img_id ${SLURM_ARRAY_TASK_ID}