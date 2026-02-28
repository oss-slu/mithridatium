import subprocess
import sys
from pathlib import Path

import torch
from mithridatium.loader import build_model


def test_demo_invisible_runs(tmp_path, monkeypatch):
    """Smoke test for the demo script.

    We create a trivial model and a zero UAP, then invoke the script via
    ``python -m scripts.demo_invisible_trigger``.  The goal is simply to
    ensure the command-line interface works and the script returns exit code
    zero; we do not check the output files.
    """
    model_file = tmp_path / "model.pth"
    uap_file = tmp_path / "uap.pt"

    # save random model parameters
    model, _ = build_model("resnet18", num_classes=10)
    torch.save(model.state_dict(), model_file)

    # save a trivial UAP so the script can load it
    torch.save(torch.zeros((3, 32, 32)), uap_file)

    cmd = [sys.executable, "-m", "scripts.demo_invisible_trigger",
           "--model", str(model_file),
           "--uap-path", str(uap_file),
           "--num-images", "2",
           "--target-class", "0",
           "--save-dir", str(tmp_path / "out")]

    # Run and assert it exits cleanly
    res = subprocess.run(cmd, cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert res.returncode == 0, f"script failed: {res.stdout}\n{res.stderr}"
