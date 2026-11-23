#!/bin/bash
# Train the #190 benchmark set: one ResNet-18 per attack, each recorded in
# models/benchmark.json (path, attack, target, poison rate, trigger, CA, ASR).
#
#   sbatch scripts/train_benchmark.sbatch                 # all models
#   sbatch --array=3 scripts/train_benchmark.sbatch       # just one (index below)
#
# Label-consistent needs its harder-image corpus first:
#   python -m scripts.fetch_label_consistent --corpus two_300
#
# Recipe: weight decay and a cosine schedule, opt-in flags in train_resnet18.py.
# The original recipe (constant lr 0.1, no weight decay) gave the existing
# models about 58% clean accuracy. The clean reference is retrained with the
# same recipe so "CA >= 90% of the clean model's" compares like with like.
#SBATCH --job-name=mith-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/benchmark_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p logs models/benchmark

RECIPE="--epochs 100 --lr 0.1 --lr_schedule cosine --weight_decay 5e-4 --seed 1 --target_class 0"

NAMES=(clean badnet_10 badnet_1 blend wanet label_consistent semantic invisible)
ARGS=(
  "--dataset clean"
  "--dataset poison --train_poison_rate 0.1"
  "--dataset poison --train_poison_rate 0.01"
  "--dataset blend --train_poison_rate 0.1 --blend-alpha 0.2"
  "--dataset wanet --train_poison_rate 0.1 --wanet-k 4 --wanet-s 0.5 --wanet-cross-ratio 2"
  "--dataset label_consistent --lc-images data/label_consistent/two_300.npy --lc-proportion 1.0 --lc-trigger all-corners"
  "--dataset semantic --train_poison_rate 0.1"
  "--dataset invisible --train_poison_rate 0.1 --uap-path models/benchmark/invisible_uap.pt"
)

i=${SLURM_ARRAY_TASK_ID:-0}
name=${NAMES[$i]}
echo "training benchmark model $name: ${ARGS[$i]}"
python scripts/train_resnet18.py $RECIPE ${ARGS[$i]} \
  --output_path "models/benchmark/resnet18_${name}.pth" \
  --benchmark-name "$name"
