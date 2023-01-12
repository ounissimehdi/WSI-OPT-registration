#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=300:00:00
#SBATCH --chdir=.
#SBATCH --output=./bash-log/%A_%a.txt
#SBATCH --error=./bash-log/%A_%a.txt
#SBATCH --job-name=reg-LARGE

python reg_large.py