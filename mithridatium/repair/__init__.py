from torchvision import datasets
import json
import shutil

from datetime import datetime, timezone 

def repair_lmr_stub(model : str, out: str, report: str, dataset:str):
    shutil.copy2(model,out)

    dummy_report = {
        "mithridatium_version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model),
        "defense": "lmr",
        "dataset": dataset,
        "results": {"status": "stub_complete"},
    }

    with open(report, "w") as f:
        json.dump(dummy_report, f, indent=2)