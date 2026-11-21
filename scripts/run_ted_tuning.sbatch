#!/bin/bash
#SBATCH --job-name=mith-ted-tuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=ted_tuning_%j.out

cd ~/mithridatium
source .venv/bin/activate
nvidia-smi

MITHRIDATIUM_NUM_WORKERS=0 python -m scripts.tune_ted \
  --models models/resnet18_clean.pth models/resnet18_poison.pth \
  --num-samples 256 --seeds 42 43 44 \
  --output-dir reports/ted_tuning_gpu --device cuda
