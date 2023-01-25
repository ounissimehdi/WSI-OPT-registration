#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/%A_%a.txt
#SBATCH --error=./bash-log/%A_%a.txt
#SBATCH --job-name=S1:reg-para
#SBATCH --array=1-219

python para_reg_v3.py -img_id ${SLURM_ARRAY_TASK_ID}