"""
Integration test for scripts.demo_invisible_trigger.

This checks that the invisible trigger demo can be invoked as a module with a
temporary checkpoint and temporary UAP file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from mithridatium.loader import build_model


def test_demo_invisible_trigger_script_runs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]

    model_file = tmp_path / "model.pth"
    uap_file = tmp_path / "uap.pt"
    save_dir = tmp_path / "out"

    model, _ = build_model("resnet18_cifar", num_classes=10)
    torch.save(model.state_dict(), model_file)

    torch.save(torch.zeros((3, 32, 32)), uap_file)

    cmd = [
        sys.executable,
        "-m",
        "scripts.demo_invisible_trigger",
        "--model",
        str(model_file),
        "--uap-path",
        str(uap_file),
        "--num-images",
        "2",
        "--target-class",
        "0",
        "--save-dir",
        str(save_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        "demo_invisible_trigger failed\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )

    assert save_dir.exists()