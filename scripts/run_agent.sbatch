#!/bin/bash
#SBATCH --job-name=mith-agent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/agent_%j.out

cd ~/mithridatium
source .venv/bin/activate
mithridatium agent -a detect "Run TED on models/resnet18_poison.pth"
mithridatium agent -a detect "Run SCALE-UP and TED on models/resnet18_clean.pth"
mithridatium agent -a repair "Run LMR on models/resnet18_poison.pth with target class 0 and prune ratio 0.05"
mithridatium agent -a audit "Run STRIP and MMBD on models/resnet18_clean.pth"
mithridatium agent "Check models/resnet18_poison.pth and repair it if it's poisoned"
