from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from mithridatium import utils
from mithridatium.defenses.mmbd import get_device


def _to_nchw_float_tensor(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    x = torch.as_tensor(data, dtype=torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape={tuple(x.shape)}")
    if x.shape[1] != 3 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


def _nan_to_none_list(arr: np.ndarray) -> list:
    out = arr.astype(object)
    out[np.isnan(arr)] = None
    return out.tolist()


def _sanitize_cache_component(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return token or "model"


def _cache_namespace_from_model_path(model_path: str | Path) -> str:
    path = Path(model_path)
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        resolved = path

    hasher = hashlib.sha1()
    hasher.update(str(resolved).encode("utf-8"))
    try:
        stat = path.stat()
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
    except OSError:
        pass

    digest = hasher.hexdigest()[:12]
    return f"{_sanitize_cache_component(path.stem)}-{digest}"


def _normalized_bounds(configs: utils.PreprocessConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(configs.get_mean(), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(configs.get_std(), device=device, dtype=torch.float32).view(1, 3, 1, 1)
    clip_min = (0.0 - mean) / std
    clip_max = (1.0 - mean) / std
    return clip_min, clip_max


def _clip_image(image: np.ndarray, clip_min: float | np.ndarray, clip_max: float | np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(clip_min, image), clip_max)


def _compute_distance(x_ori: np.ndarray, x_pert: np.ndarray, constraint: str = "l2") -> float:
    if constraint == "l2":
        return float(np.linalg.norm(x_ori - x_pert))
    if constraint == "linf":
        return float(np.max(np.abs(x_ori - x_pert)))
    raise ValueError(f"Unsupported constraint '{constraint}'")


def _predict_labels(
    model: torch.nn.Module,
    images: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    x = _to_nchw_float_tensor(images)
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i : i + batch_size].to(device=device, dtype=torch.float32)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def decision_function(
    model: torch.nn.Module,
    data: torch.Tensor | np.ndarray,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)
    return _predict_labels(model, data, device=device, batch_size=batch_size)


def _hsja_decision_function(model: torch.nn.Module, images: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    images = _clip_image(images, params["clip_min"], params["clip_max"])
    preds = _predict_labels(
        model,
        images,
        device=params["device"],
        batch_size=params["query_batch_size"],
    )
    if params["target_label"] is None:
        return preds != params["original_label"]
    return preds == params["target_label"]


def _approximate_gradient(
    model: torch.nn.Module,
    sample: np.ndarray,
    num_evals: int,
    delta: float,
    params: Dict[str, Any],
) -> np.ndarray:
    noise_shape = (num_evals, *params["shape"])
    if params["constraint"] == "l2":
        rv = np.random.randn(*noise_shape).astype(np.float32)
    else:
        rv = np.random.uniform(low=-1.0, high=1.0, size=noise_shape).astype(np.float32)

    rv_norm = np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))
    rv = rv / np.maximum(rv_norm, 1e-12)

    perturbed = sample + delta * rv
    perturbed = _clip_image(perturbed, params["clip_min"], params["clip_max"])
    rv = (perturbed - sample) / max(delta, 1e-12)

    decisions = _hsja_decision_function(model, perturbed, params)
    decision_shape = (len(decisions), 1, 1, 1)
    fval = 2.0 * decisions.astype(np.float32).reshape(decision_shape) - 1.0

    fmean = float(np.mean(fval))
    if fmean == 1.0:
        gradf = np.mean(rv, axis=0)
    elif fmean == -1.0:
        gradf = -np.mean(rv, axis=0)
    else:
        fval = fval - fmean
        gradf = np.mean(fval * rv, axis=0)

    grad_norm = float(np.linalg.norm(gradf))
    if grad_norm < 1e-12:
        return gradf
    return gradf / grad_norm


def _project(
    original_image: np.ndarray,
    perturbed_images: np.ndarray,
    alphas: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    alphas = alphas.reshape((len(alphas), 1, 1, 1))
    if params["constraint"] == "l2":
        return (1.0 - alphas) * original_image + alphas * perturbed_images
    out_images = _clip_image(perturbed_images, original_image - alphas, original_image + alphas)
    return out_images


def _binary_search_batch(
    original_image: np.ndarray,
    perturbed_images: np.ndarray,
    model: torch.nn.Module,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    dists_post_update = np.array(
        [_compute_distance(original_image, p, params["constraint"]) for p in perturbed_images],
        dtype=np.float32,
    )

    if params["constraint"] == "linf":
        highs = dists_post_update.copy()
        thresholds = np.minimum(dists_post_update * params["theta"], params["theta"])
    else:
        highs = np.ones(len(perturbed_images), dtype=np.float32)
        thresholds = np.full(len(perturbed_images), params["theta"], dtype=np.float32)
    lows = np.zeros(len(perturbed_images), dtype=np.float32)

    thresholds = np.maximum(thresholds, 1e-12)

    while np.max((highs - lows) / thresholds) > 1.0:
        mids = (highs + lows) / 2.0
        mid_images = _project(original_image, perturbed_images, mids, params)
        decisions = _hsja_decision_function(model, mid_images, params)
        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = _project(original_image, perturbed_images, highs, params)
    dists = np.array(
        [_compute_distance(original_image, img, params["constraint"]) for img in out_images],
        dtype=np.float32,
    )
    idx = int(np.argmin(dists))
    return out_images[idx], float(dists_post_update[idx])


def _initialize(model: torch.nn.Module, sample: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    if params["target_image"] is not None:
        return params["target_image"].astype(np.float32)

    num_evals = 0
    while True:
        random_noise = np.random.uniform(params["clip_min"], params["clip_max"], size=params["shape"]).astype(np.float32)
        success = bool(_hsja_decision_function(model, random_noise[None], params)[0])
        num_evals += 1
        if success:
            break
        if num_evals >= 10_000:
            raise RuntimeError("Initialization failed after 10k tries. Provide a target_image for targeted HSJA.")

    low = 0.0
    high = 1.0
    while high - low > 1e-3:
        mid = (high + low) / 2.0
        blended = (1.0 - mid) * sample + mid * random_noise
        success = bool(_hsja_decision_function(model, blended[None], params)[0])
        if success:
            high = mid
        else:
            low = mid

    return ((1.0 - high) * sample + high * random_noise).astype(np.float32)


def _geometric_progression_for_stepsize(
    x: np.ndarray,
    update: np.ndarray,
    dist: float,
    model: torch.nn.Module,
    params: Dict[str, Any],
) -> float:
    epsilon = dist / np.sqrt(max(params["cur_iter"], 1))
    epsilon = float(max(epsilon, 1e-12))

    for _ in range(60):
        new = x + epsilon * update
        success = bool(_hsja_decision_function(model, new[None], params)[0])
        if success:
            return epsilon
        epsilon /= 2.0
    return epsilon


def _select_delta(params: Dict[str, Any], dist_post_update: float) -> float:
    if params["cur_iter"] == 1:
        return 0.1 * params["clip_range"]
    if params["constraint"] == "l2":
        return np.sqrt(params["d"]) * params["theta"] * dist_post_update
    return params["d"] * params["theta"] * dist_post_update


def hsja(
    model: torch.nn.Module,
    sample: np.ndarray,
    *,
    clip_max: float | np.ndarray,
    clip_min: float | np.ndarray,
    constraint: str = "l2",
    num_iterations: int = 50,
    gamma: float = 1.0,
    target_label: Optional[int] = None,
    target_image: Optional[np.ndarray] = None,
    stepsize_search: str = "geometric_progression",
    max_num_evals: int = 30_000,
    init_num_evals: int = 100,
    query_batch_size: int = 512,
    verbose: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[float, np.ndarray]:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    original_label = int(_predict_labels(model, sample[None], device=device, batch_size=query_batch_size)[0])
    params: Dict[str, Any] = {
        "clip_max": clip_max,
        "clip_min": clip_min,
        "clip_range": float(np.max(np.asarray(clip_max) - np.asarray(clip_min))),
        "shape": sample.shape,
        "original_label": original_label,
        "target_label": target_label,
        "target_image": target_image,
        "constraint": constraint,
        "num_iterations": int(num_iterations),
        "gamma": float(gamma),
        "d": int(np.prod(sample.shape)),
        "stepsize_search": stepsize_search,
        "max_num_evals": int(max_num_evals),
        "init_num_evals": int(init_num_evals),
        "query_batch_size": int(query_batch_size),
        "verbose": bool(verbose),
        "device": device,
    }

    if constraint == "l2":
        params["theta"] = params["gamma"] / (np.sqrt(params["d"]) * params["d"])
    else:
        params["theta"] = params["gamma"] / (params["d"] ** 2)

    perturbed = _initialize(model, sample, params)
    perturbed, dist_post_update = _binary_search_batch(sample, perturbed[None], model, params)
    dist = _compute_distance(perturbed, sample, constraint)

    for j in range(params["num_iterations"]):
        params["cur_iter"] = j + 1
        delta = float(_select_delta(params, dist_post_update))
        num_evals = int(params["init_num_evals"] * np.sqrt(j + 1))
        num_evals = int(min(num_evals, params["max_num_evals"]))

        gradf = _approximate_gradient(model, perturbed, num_evals, delta, params)
        update = np.sign(gradf) if constraint == "linf" else gradf

        if params["stepsize_search"] == "geometric_progression":
            epsilon = _geometric_progression_for_stepsize(perturbed, update, dist, model, params)
            perturbed = _clip_image(perturbed + epsilon * update, clip_min, clip_max)
            perturbed, dist_post_update = _binary_search_batch(sample, perturbed[None], model, params)
        else:
            epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
            perturbeds = perturbed + epsilons.reshape((20, 1, 1, 1)) * update
            perturbeds = _clip_image(perturbeds, clip_min, clip_max)
            idx_perturbed = _hsja_decision_function(model, perturbeds, params)
            if np.sum(idx_perturbed) > 0:
                perturbed, dist_post_update = _binary_search_batch(sample, perturbeds[idx_perturbed], model, params)

        dist = _compute_distance(perturbed, sample, constraint)
        if verbose:
            print(f"[AEVA/HSJA] iteration: {j + 1}, distance {dist:.4E}")

    return dist, perturbed


def attack(
    model: torch.nn.Module,
    basic_imgs: torch.Tensor,
    target_imgs: torch.Tensor,
    target_labels: torch.Tensor,
    *,
    clip_max: float | np.ndarray,
    clip_min: float | np.ndarray,
    num_iterations: int = 50,
    max_num_evals: int = 30_000,
    init_num_evals: int = 100,
    query_batch_size: int = 512,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    basic_np = basic_imgs.detach().cpu().numpy().astype(np.float32)
    target_np = target_imgs.detach().cpu().numpy().astype(np.float32)
    target_lab = target_labels.detach().cpu().numpy().astype(np.int64)

    n = basic_np.shape[0]
    vec = np.empty((n,), dtype=np.float32)
    vec_per = np.empty_like(basic_np, dtype=np.float32)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = get_device(0)

    for i in range(n):
        sample = basic_np[i]
        target_image = target_np[i]
        target_label = int(target_lab[i])

        dist, perturbed = hsja(
            model,
            sample,
            clip_max=clip_max,
            clip_min=clip_min,
            constraint="l2",
            num_iterations=num_iterations,
            gamma=1.0,
            target_label=target_label,
            target_image=target_image,
            stepsize_search="geometric_progression",
            max_num_evals=max_num_evals,
            init_num_evals=init_num_evals,
            query_batch_size=query_batch_size,
            verbose=verbose,
            device=device,
        )
        per = perturbed - sample
        vec[i] = float(np.max(np.abs(per)) / max(dist, 1e-12))
        vec_per[i] = per

    return vec, vec_per


def _pair_gap_peak(perturbations_abs: np.ndarray) -> float:
    if perturbations_abs.size == 0:
        return float("nan")

    n = perturbations_abs.shape[0]
    flat_line_sums = np.sum(perturbations_abs, axis=-1).reshape(n, -1)
    top2 = np.sort(flat_line_sums, axis=1)[:, -2:].sum(axis=1)
    denom = perturbations_abs.reshape(n, -1).sum(axis=1)
    score = np.divide(top2, denom, out=np.zeros_like(top2), where=denom > 0)
    return float(np.max(score))


class BDDetect:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        task: str = "cifar10",
        configs: Optional[utils.PreprocessConfig] = None,
        device: Optional[torch.device] = None,
        output_dir: Optional[str | Path] = None,
        cache_namespace: Optional[str] = None,
        batch_size: int = 256,
        samples_per_class: int = 40,
        hsja_iterations: int = 50,
        hsja_max_num_evals: int = 30_000,
        hsja_init_num_evals: int = 100,
        hsja_query_batch_size: int = 512,
        anomaly_index_threshold: float = 4.0,
        verbose: bool = False,
    ):
        task = task.lower().strip()
        if task not in {"cifar10", "cifar100", "cifar10_for_imagenet"}:
            raise ValueError("AEVA currently supports task in {'cifar10', 'cifar100', 'cifar10_for_imagenet'} for this repo.")

        self.task = task
        self.configs = configs or utils.get_preprocess_config(task)
        self.num_labels = int(self.configs.get_num_classes())        
        self.device = device or get_device(0)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()

        self.samples_per_class = int(samples_per_class)
        self.hsja_iterations = int(hsja_iterations)
        self.hsja_max_num_evals = int(hsja_max_num_evals)
        self.hsja_init_num_evals = int(hsja_init_num_evals)
        self.hsja_query_batch_size = int(hsja_query_batch_size)
        self.anomaly_index_threshold = float(anomaly_index_threshold)
        self.verbose = bool(verbose)

        default_dir = Path(f"{task}_adv_per")
        if output_dir is None:
            namespace = _sanitize_cache_component(cache_namespace or "model")
            self.output_dir = default_dir / namespace
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        clip_min_t, clip_max_t = _normalized_bounds(self.configs, self.device)
        self.clip_min = clip_min_t.squeeze(0).detach().cpu().numpy()
        self.clip_max = clip_max_t.squeeze(0).detach().cpu().numpy()

        self.x_val, self.y_val, self.clean_accuracy = self._collect_correct_samples(batch_size=batch_size)

    def _collect_correct_samples(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        loader, _ = utils.dataloader_for(self.task, split="test", batch_size=batch_size)

        x_keep, y_keep = [], []
        total = 0
        correct = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device)
                pred = self.model(x).argmax(dim=1)
                mask = pred == y
                if mask.any():
                    x_keep.append(x[mask].cpu())
                    y_keep.append(y[mask].cpu())
                correct += int(mask.sum().item())
                total += int(y.numel())

        if not x_keep:
            raise RuntimeError("No correctly classified validation samples found for AEVA.")

        x_val = torch.cat(x_keep, dim=0)
        y_val = torch.cat(y_keep, dim=0)
        return x_val, y_val, float(correct / max(total, 1))

    def get_vec(self, original_label: int, target_label: int) -> Tuple[Optional[float], Optional[float]]:
        out_path = self.output_dir / f"data_{original_label}_{target_label}.npy"

        if out_path.exists():
            per = np.load(out_path)
            if per.size == 0:
                return None, None
            peak = _pair_gap_peak(np.abs(per))
            mean_l2 = float(np.mean(np.linalg.norm(per.reshape(per.shape[0], -1), axis=1)))
            return peak, mean_l2

        x_o = self.x_val[self.y_val == original_label][: self.samples_per_class]
        x_t = self.x_val[self.y_val == target_label][: self.samples_per_class]
        y_t = self.y_val[self.y_val == target_label][: self.samples_per_class]

        n = int(min(len(x_o), len(x_t), len(y_t)))
        if n == 0:
            return None, None

        x_o = x_o[:n].to(device=self.device, dtype=torch.float32)
        x_t = x_t[:n].to(device=self.device, dtype=torch.float32)
        y_t = y_t[:n].to(device=self.device, dtype=torch.long)

        _, per = attack(
            self.model,
            x_o,
            x_t,
            y_t,
            clip_max=self.clip_max,
            clip_min=self.clip_min,
            num_iterations=self.hsja_iterations,
            max_num_evals=self.hsja_max_num_evals,
            init_num_evals=self.hsja_init_num_evals,
            query_batch_size=self.hsja_query_batch_size,
            verbose=self.verbose,
        )
        np.save(out_path, per)

        peak = _pair_gap_peak(np.abs(per))
        mean_l2 = float(np.mean(np.linalg.norm(per.reshape(per.shape[0], -1), axis=1)))
        return peak, mean_l2

    def _load_pair_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        pair_gap = np.full((self.num_labels, self.num_labels), np.nan, dtype=np.float32)
        pair_mean_l2 = np.full((self.num_labels, self.num_labels), np.nan, dtype=np.float32)

        for source_label in range(self.num_labels):
            for target_label in range(self.num_labels):
                if source_label == target_label:
                    continue
                path = self.output_dir / f"data_{source_label}_{target_label}.npy"
                if not path.exists():
                    continue
                per = np.load(path)
                if per.size == 0:
                    continue
                pair_gap[source_label, target_label] = _pair_gap_peak(np.abs(per))
                pair_mean_l2[source_label, target_label] = float(
                    np.mean(np.linalg.norm(per.reshape(per.shape[0], -1), axis=1))
                )

        return pair_gap, pair_mean_l2

    def detect(self, sp: int = 0, ep: Optional[int] = None) -> Dict[str, Any]:
        if ep is None:
            ep = self.num_labels
        if not (0 <= sp < ep <= self.num_labels):
            raise ValueError(f"Invalid class range: sp={sp}, ep={ep}, num_labels={self.num_labels}")

        for source_label in range(sp, ep):
            labels = list(range(self.num_labels))
            labels.remove(source_label)
            for target_label in labels:
                self.get_vec(original_label=source_label, target_label=target_label)

        pair_gap, pair_mean_l2 = self._load_pair_matrices()
        valid_counts = np.sum(~np.isnan(pair_gap), axis=0)
        gap = np.where(valid_counts > 0, np.nansum(pair_gap, axis=0), np.nan)

        finite = np.isfinite(gap)
        if finite.sum() == 0:
            raise RuntimeError("No AEVA pair scores available. Run detect on at least one source class.")

        median = float(np.median(gap[finite]))
        mad = float(1.4826 * np.median(np.abs(gap[finite] - median)))
        if mad < 1e-12:
            mad = 1e-12

        anomaly_index = np.full_like(gap, np.nan, dtype=np.float32)
        anomaly_index[finite] = (gap[finite] - median) / mad

        suspected_target = int(np.nanargmax(anomaly_index))
        suspicion_score = float(anomaly_index[suspected_target])
        suspected_backdoor = bool(suspicion_score >= self.anomaly_index_threshold)
        verdict = "likely backdoored" if suspected_backdoor else "likely clean"

        return {
            "defense": "aeva",
            "dataset": self.task,
            "clean_accuracy": self.clean_accuracy,
            "verdict": verdict,
            "suspected_backdoor": suspected_backdoor,
            "num_flagged": int(suspected_backdoor),
            "suspected_target": suspected_target if suspected_backdoor else None,
            "suspicion_score": suspicion_score,
            "top_eigenvalue": suspicion_score,
            "global_adversarial_peak": _nan_to_none_list(gap),
            "anomaly_index": _nan_to_none_list(anomaly_index),
            "pair_gap_scores": _nan_to_none_list(pair_gap),
            "source_target_mean_l2": _nan_to_none_list(pair_mean_l2),
            "thresholds": {
                "anomaly_index_threshold": self.anomaly_index_threshold,
            },
            "parameters": {
                "sp": sp,
                "ep": ep,
                "samples_per_class": self.samples_per_class,
                "hsja_iterations": self.hsja_iterations,
                "hsja_max_num_evals": self.hsja_max_num_evals,
                "hsja_init_num_evals": self.hsja_init_num_evals,
                "hsja_query_batch_size": self.hsja_query_batch_size,
                "device": str(self.device),
            },
            "output_dir": str(self.output_dir),
        }


BD_detect = BDDetect


def run_aeva(
    model: torch.nn.Module,
    configs: Optional[utils.PreprocessConfig] = None,
    *,
    task: Optional[str] = None,
    model_path: Optional[str | Path] = None,
    sp: int = 0,
    ep: Optional[int] = None,
    device: Optional[torch.device] = None,
    output_dir: Optional[str | Path] = None,
    batch_size: int = 256,
    samples_per_class: int = 40,
    hsja_iterations: int = 50,
    hsja_max_num_evals: int = 30_000,
    hsja_init_num_evals: int = 100,
    hsja_query_batch_size: int = 512,
    anomaly_index_threshold: float = 4.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    if task is None:
        if configs is not None:
            task = str(configs.get_dataset())
        else:
            task = "cifar10"

    cache_namespace = None
    if output_dir is None and model_path is not None:
        cache_namespace = _cache_namespace_from_model_path(model_path)

    detector = BDDetect(
        model=model,
        task=task,
        configs=configs,
        device=device,
        output_dir=output_dir,
        cache_namespace=cache_namespace,
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        hsja_iterations=hsja_iterations,
        hsja_max_num_evals=hsja_max_num_evals,
        hsja_init_num_evals=hsja_init_num_evals,
        hsja_query_batch_size=hsja_query_batch_size,
        anomaly_index_threshold=anomaly_index_threshold,
        verbose=verbose,
    )
    return detector.detect(sp=sp, ep=ep)
