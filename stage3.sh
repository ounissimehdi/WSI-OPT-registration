#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/REG-%A_%a.txt
#SBATCH --error=./bash-log/REG-%A_%a.txt
#SBATCH --job-name=S3:reg-LARGE

python gif_creation.py