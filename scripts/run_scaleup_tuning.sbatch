#!/bin/bash
#SBATCH --job-name=mith-scaleup-tuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/scaleup_tuning_%j.out

cd ~/mithridatium
source .venv/bin/activate
MITHRIDATIUM_NUM_WORKERS=0 python -m scripts.tune_scaleup
