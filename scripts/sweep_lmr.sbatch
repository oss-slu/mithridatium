#!/bin/bash
#SBATCH --job-name=mith-lmr-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/lmr_sweep_%j.out

cd ~/mithridatium
source .venv/bin/activate
python scripts/sweep_lmr.py

