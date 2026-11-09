#!/bin/bash
#SBATCH --job-name=mith-ted-lmr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ted_lmr_%j.out

cd ~/mithridatium
source .venv/bin/activate
nvidia-smi

mithridatium detect --model models/resnet18_poison.pth --method ted \
  --data cifar10 --out reports/detect_ted_poison.json --force
mithridatium detect --model models/resnet18_clean.pth --method ted \
  --data cifar10 --out reports/detect_ted_clean.json --force

mithridatium repair --model models/resnet18_poison.pth --method lmr --data cifar10 \
  --clean-samples 500 --seed 42 --lmr-target-class 0 --lmr-prune-ratio 0.10 \
  --out models/resnet18_lmr.pth --report reports/repair_lmr.json --force
