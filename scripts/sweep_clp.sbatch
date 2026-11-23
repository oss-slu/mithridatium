#!/bin/bash
#SBATCH --job-name=mith-clp-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clp_sweep_%j.out

cd ~/mithridatium
source .venv/bin/activate
python -m scripts.sweep_clp
