# from __future__ import absolute_import
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# import argparse
# import argparse
import random
import numpy as np

#Code adapted from https://github.com/wanghangpsu/MM-BD/blob/main/univ_bd.py

NUM_IMAGES = 30
NSTEP = 75


def get_device(device_index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_index}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# parser = argparse.ArgumentParser(description='UnivBD method')
# parser.add_argument('--model_dir', default='model1', help='model path')
# parser.add_argument('--device', default=0, type=int)
# parser.add_argument("--report_out", default="reports/mmbd_report.json", help="JSON output path")
#parser.add_argument('--data_path', '-d', required=True, help='data path')
# args = parser.parse_args()
# parser = argparse.ArgumentParser(description='UnivBD method')
# parser.add_argument('--model_dir', default='model1', help='model path')
# parser.add_argument('--device', default=0, type=int)
# parser.add_argument("--report_out", default="reports/mmbd_report.json", help="JSON output path")
# parser.add_argument('--data_path', '-d', required=True, help='data path')
# args = parser.parse_args()

'''def load_resnet18_cifar10(weights_path, device=0):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)

    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model'''


def run_mmbd(model, configs, device=None, classes=None):

    random.seed()
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    # Detection parameters
    try:
        NC = int(getattr(model, "num_classes"))
    except Exception:
        NC = None

    if NC is None and hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        NC = int(model.fc.out_features)

    if NC is None:
        raise ValueError(
            "MMBD could not determine the model's number of output classes."
    )

    NI = 150
    PI = 0.9
    TC = 6
    batch_size = 20

    # Resolve which classes to probe
    classes_to_probe = classes
    if classes_to_probe is None and hasattr(configs, "mmbd_classes"):
        classes_to_probe = getattr(configs, "mmbd_classes")

    if classes_to_probe is None:
        N_CLASSES_TO_PROBE = min(5, NC)
        classes_to_probe = list(range(N_CLASSES_TO_PROBE))
    else:
        classes_to_probe = list(classes_to_probe)

    for c in classes_to_probe:
        if isinstance(c, bool) or not isinstance(c, (int, np.integer)):
            raise ValueError(f"Invalid class index '{c}': must be an integer.")
        if c < 0 or c >= NC:
            raise ValueError(f"Class index {c} is out of range for model with {NC} classes.")
    if len(classes_to_probe) != len(set(classes_to_probe)):
        raise ValueError("Duplicate class indices found in MMBD classes to probe.")
    if len(classes_to_probe) == 0:
        raise ValueError("MMBD requires at least one class to probe.")

    # Load model
    model = model.to(device=device, dtype=torch.float32).eval()
    criterion = nn.CrossEntropyLoss()

    
    model = model.to(device).eval()
    mean = torch.tensor(configs.get_mean(), device=device).view(1, 3, 1, 1)
    std = torch.tensor(configs.get_std(), device=device).view(1, 3, 1, 1)

    def lr_scheduler(iter_idx):
        lr = 1e-2


        return lr

    res = []
    num_probed = len(classes_to_probe)
    for idx, t in enumerate(classes_to_probe):
        print(f"[MMBD] optimizing class {idx+1}/{num_probed} (class {t})…", flush=True)
        images = torch.rand([NUM_IMAGES, *configs.input_size], device=device, dtype=torch.float32, requires_grad=True)
        last_loss = 1000.0
        labels = torch.full((len(images),), t, dtype=torch.long, device=device)
        onehot_label = F.one_hot(labels, num_classes=NC).to(device=device, dtype=torch.float32)

        optimizer = torch.optim.SGD([images], lr=1e-2, momentum=0.9)
        

        for iter_idx in range(NSTEP):
            optimizer.zero_grad(set_to_none=True)

            x = torch.clamp(images, 0, 1)
            x = (x - mean) / std
            outputs = model(x)

            loss = (-(outputs * onehot_label).sum()
                    + torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1).values.sum())
            loss.backward()
            optimizer.step()

            curr = float(loss.item())
            if iter_idx % 50 == 0 or iter_idx == NSTEP - 1:
                print(f"[MMBD]   Iter {iter_idx}/{NSTEP}, loss={curr:.4f}")
            if abs(last_loss - curr) / max(abs(last_loss), 1e-12) < 1e-5:
                print(f"[MMBD]   Converged early at iter {iter_idx}")
                break
            last_loss = curr

        res.append(torch.max(torch.sum(outputs * onehot_label, dim=1)
                - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1).values).item())

    stats = np.array(res, dtype=float)
    from scipy.stats import median_abs_deviation as MAD
    from scipy.stats import gamma
    mad = MAD(stats, scale='normal')
    mad = float(mad) if mad != 0 else 1e-12
    abs_deviation = np.abs(stats - np.median(stats))
    score = abs_deviation / mad


    ind_max = int(np.argmax(stats))
    suspected_target = classes_to_probe[ind_max]
    r_eval = float(np.amax(stats))
    r_null = np.delete(stats, ind_max)

    pv = 1.0
    if len(r_null) > 1:
        try:
            shape, loc, scale = gamma.fit(r_null)
            pv = float(1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null)+1))
        except Exception:
            pv = 1.0

    verdict = "Likely clean" if pv > 0.05 else "Likely backdoored"

    # suspected_backdoor = (verdict == "attack")
    # num_flagged = 1 if suspected_backdoor else 0
    top_eigenvalue = float(r_eval)


    thresholds = {
        "p_value": 0.05,
        "normalized_score": {
            "normal": [0.0, 1.5],
            "mild": [1.5, 3.0],
            "suspicious": [3.0, 5.0],
            "very_suspicious": [5.0, None]
        },
    }

    parameters = {
        "NC": NC,
        "NSTEP": NSTEP,
        "optimizer": "SGD(momentum=0.2)",
        "lr_init": 1e-2,
        "device": str(device),
        "classes_probed": classes_to_probe,
    }

    results = {
        "defense": "mmbd",
        "per_class_scores": stats.tolist(),
        "normalized_scores": score.tolist(),
        "classes_probed": classes_to_probe,
        "p_value": float(pv),
        "verdict": verdict,
        "suspected_target": (int(suspected_target) if verdict == "Likely backdoored" else None),
        "thresholds": thresholds,
        "parameters": parameters,
        "dataset": configs.get_dataset(),

        # "suspected_backdoor": suspected_backdoor,
        # "num_flagged": int(num_flagged),
        "top_eigenvalue": float(top_eigenvalue),
    }

    return results

    '''build_report(
        model_path=args.model_dir,
        defense="MMBD",
        out_path=args.report_out,
        details=results,
        version="0.1.1"
    )'''