#!/bin/bash
#SBATCH --job-name=mith-anp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/anp_%j.out

cd ~/mithridatium
source .venv/bin/activate
nvidia-smi
git branch --show-current

# ANP unit, smoke, and report schema tests
python -m pytest -q tests/unit/test_anp.py tests/smoke/test_repair_anp.py \
  tests/integration/test_anp_report_schema.py

ASR="--anp-attack-target 0 --anp-trigger-type white-square --anp-trigger-size 4 --anp-trigger-pos bottom-right"

# threshold mode (default 0.2)
mithridatium repair --model models/resnet18_poison.pth --method anp --data cifar10 \
  --clean-samples 500 --seed 42 $ASR \
  --out models/resnet18_anp_threshold.pth --report reports/repair_anp_threshold.json --force

# fraction mode
mithridatium repair --model models/resnet18_poison.pth --method anp --data cifar10 \
  --clean-samples 500 --seed 42 --anp-prune-fraction 0.05 $ASR \
  --out models/resnet18_anp_frac05.pth --report reports/repair_anp_frac05.json --force

# fraction 0 used to crash; should prune nothing and leave accuracy/ASR unchanged
mithridatium repair --model models/resnet18_poison.pth --method anp --data cifar10 \
  --clean-samples 500 --seed 42 --anp-prune-fraction 0 $ASR \
  --out models/resnet18_anp_frac0.pth --report reports/repair_anp_frac0.json --force

# MCP path (the run_anp tool used to crash on the missing parameters)
python - <<'EOF'
from mithridatium.mcp.repair_mcp import _repair_kwargs, _run_repair
result = _run_repair(_repair_kwargs(
    "anp", model="models/resnet18_poison.pth", out="models/resnet18_anp_mcp.pth",
    force=True, seed=42, anp_iterations=200, anp_prune_fraction=0.05,
    anp_attack_target=0, anp_trigger_type="white-square",
    anp_trigger_size=4, anp_trigger_pos="bottom-right",
))
print("MCP ok:", result["report_path"])
EOF
