"""TED++ input-level backdoor detection.

TED++ extends TED's layer-wise nearest-neighbour ranks with class-specific
activation tubes and scores the resulting rank vectors with PCA reconstruction
error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, Subset


def _collect_activations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Return labels, flattened hooked activations, and predictions in loader order."""
    activations: dict[str, torch.Tensor] = {}
    collected: dict[str, list[torch.Tensor]] = {}
    labels_all: list[torch.Tensor] = []
    predictions_all: list[torch.Tensor] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []
    layer_names: dict[nn.Module, str] = {}
    layer_index = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
            layer_names[module] = name or f"Conv2d_{layer_index}"
            layer_index += 1
        elif isinstance(module, nn.ReLU):
            layer_names[module] = name or f"ReLU_{layer_index}"
            layer_index += 1
        elif isinstance(module, nn.Linear):
            layer_names[module] = name or f"Linear_{layer_index}"
            layer_index += 1

    if not layer_names:
        raise ValueError(
            "TED++ could not find hookable Conv2d, ReLU, or Linear layers "
            "in the supplied model."
        )

    def save_activation(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    f"TED++ expected tensor output from hooked layer '{name}', "
                    f"got {type(output).__name__}."
                )
            activations[name] = output.detach().flatten(start_dim=1)

        return hook

    for module, name in layer_names.items():
        handles.append(module.register_forward_hook(save_activation(name)))

    try:
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                activations.clear()
                outputs = model(images.to(device))
                if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
                    raise ValueError(
                        "TED++ expects the model to return logits with shape "
                        "[batch_size, num_classes]."
                    )

                batch_size = images.shape[0]
                missing_layers = set(layer_names.values()) - activations.keys()
                if missing_layers:
                    missing = ", ".join(sorted(missing_layers))
                    raise RuntimeError(
                        f"Model forward pass did not produce activations for: {missing}"
                    )
                for name, activation in activations.items():
                    if activation.shape[0] != batch_size:
                        raise ValueError(
                            f"Layer '{name}' produced {activation.shape[0]} "
                            f"activations for a batch of {batch_size}."
                        )
                    collected.setdefault(name, []).append(activation.cpu())

                labels_all.append(torch.as_tensor(labels).reshape(-1).cpu().long())
                predictions_all.append(outputs.argmax(dim=1).cpu())

        if not labels_all:
            raise ValueError("TED++ cannot process an empty dataset.")

        return (
            torch.cat(labels_all),
            {name: torch.cat(parts) for name, parts in collected.items()},
            torch.cat(predictions_all),
        )
    finally:
        for handle in handles:
            handle.remove()


def _select_clean_reference(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    activations: dict[str, torch.Tensor],
    reference_size: int,
    samples_per_class: int,
    seed: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Keep correct reference examples and cap each predicted class."""
    correct_indices = torch.where(labels == predictions)[0].numpy()
    if len(correct_indices) == 0:
        raise ValueError(
            "TED++ found no correctly classified examples in reference_data."
        )

    rng = np.random.default_rng(seed)
    if len(correct_indices) > reference_size:
        correct_indices = rng.choice(
            correct_indices, size=reference_size, replace=False
        )

    selected: list[int] = []
    predicted_numpy = predictions.numpy()
    for class_id in np.unique(predicted_numpy[correct_indices]):
        class_indices = correct_indices[predicted_numpy[correct_indices] == class_id]
        if len(class_indices) < 2:
            continue
        if len(class_indices) > samples_per_class:
            class_indices = rng.choice(
                class_indices, size=samples_per_class, replace=False
            )
        selected.extend(int(index) for index in class_indices)

    if not selected:
        raise ValueError(
            "TED++ needs at least two correctly classified reference images "
            "for one predicted class."
        )

    selected_indices = torch.tensor(selected, dtype=torch.long)
    return (
        labels[selected_indices],
        {name: values[selected_indices] for name, values in activations.items()},
        predictions[selected_indices],
    )


def _layer_tube_radius(
    reference_activations: torch.Tensor,
    reference_predictions: torch.Tensor,
    beta: float,
) -> float:
    """Compute one layer's global tube radius from clean same-class neighbours."""
    radii: list[torch.Tensor] = []
    for class_id in torch.unique(reference_predictions):
        class_activations = reference_activations[
            reference_predictions == class_id
        ]
        class_size = class_activations.shape[0]
        if class_size < 2:
            continue

        distances = torch.cdist(class_activations, class_activations)
        distances.fill_diagonal_(float("inf"))
        neighbor_count = min(math.ceil(class_size * beta), class_size - 1)
        nearest = torch.topk(
            distances, k=neighbor_count, dim=1, largest=False
        ).values
        radii.append(nearest[:, -1].max())

    if not radii:
        raise ValueError(
            "TED++ could not calculate a class tube; each reference class "
            "needs at least two examples."
        )
    return float(torch.stack(radii).max().item())


def _rank_vectors(
    query_activations: torch.Tensor,
    query_predictions: torch.Tensor,
    reference_activations: torch.Tensor,
    reference_predictions: torch.Tensor,
    tube_radius: float,
    *,
    leave_one_out: bool,
) -> np.ndarray:
    """Build a column of locally adaptive TED ranks for one layer."""
    distances = torch.cdist(query_activations, reference_activations)
    ranks: list[int] = []
    reference_count = reference_activations.shape[0]

    for query_index, query_class in enumerate(query_predictions):
        row_distances = distances[query_index].clone()
        if leave_one_out:
            row_distances[query_index] = float("inf")

        same_class = reference_predictions == query_class
        same_class_distances = row_distances[same_class]
        if same_class_distances.numel() == 0:
            ranks.append(reference_count)
            continue

        nearest_same_class = same_class_distances.min().item()
        if nearest_same_class > tube_radius:
            ranks.append(reference_count)
            continue

        ordered_indices = torch.argsort(row_distances)
        ordered_classes = reference_predictions[ordered_indices]
        matching_positions = torch.where(ordered_classes == query_class)[0]
        if matching_positions.numel() == 0:
            ranks.append(reference_count)
        else:
            ranks.append(int(matching_positions[0].item()))

    return np.asarray(ranks, dtype=np.float64)


def run_tedpp(
    model: nn.Module,
    reference_data: Dataset,
    suspect_data: Dataset,
    reference_size: int = 1000,
    contamination: float = 0.01,
    batch_size: int = 100,
    seed: int | None = None,
    device: str | torch.device = "cpu",
    beta: float = 0.5,
    samples_per_class: int = 5,
) -> dict[str, Any]:
    """Run TED++ on known-clean reference data and unknown suspect data.

    Both datasets must yield ``(image_tensor, integer_label)`` pairs. Reference
    labels are used to keep correctly classified clean examples; suspect labels
    are not used to determine whether an input is poisoned.
    """
    if reference_size < 2:
        raise ValueError("reference_size must be at least 2.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if samples_per_class < 2:
        raise ValueError("samples_per_class must be at least 2.")
    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in the interval (0, 1].")
    if not 0.0 < contamination <= 0.5:
        raise ValueError("contamination must be in the interval (0, 0.5].")
    if len(reference_data) == 0:
        raise ValueError("reference_data must not be empty.")
    if len(suspect_data) == 0:
        raise ValueError("suspect_data must not be empty.")

    device = torch.device(device)
    model.to(device)
    if len(reference_data) > reference_size:
        rng = np.random.default_rng(seed)
        reference_indices = rng.choice(
            len(reference_data), size=reference_size, replace=False
        )
        reference_data = Subset(reference_data, reference_indices.tolist())

    reference_loader = DataLoader(
        reference_data, batch_size=batch_size, shuffle=False, num_workers=0
    )
    suspect_loader = DataLoader(
        suspect_data, batch_size=batch_size, shuffle=False, num_workers=0
    )

    reference_labels, reference_activations, reference_predictions = (
        _collect_activations(model, reference_loader, device)
    )
    _, suspect_activations, suspect_predictions = _collect_activations(
        model, suspect_loader, device
    )

    reference_labels, reference_activations, reference_predictions = (
        _select_clean_reference(
            reference_labels,
            reference_predictions,
            reference_activations,
            reference_size,
            samples_per_class,
            seed,
        )
    )
    reference_count = len(reference_labels)

    reference_rank_columns: list[np.ndarray] = []
    suspect_rank_columns: list[np.ndarray] = []
    layer_radii: dict[str, float] = {}
    for layer_name, ref_layer_activations in reference_activations.items():
        radius = _layer_tube_radius(
            ref_layer_activations, reference_predictions, beta
        )
        layer_radii[layer_name] = radius
        reference_rank_columns.append(
            _rank_vectors(
                ref_layer_activations,
                reference_predictions,
                ref_layer_activations,
                reference_predictions,
                radius,
                leave_one_out=True,
            )
        )
        suspect_rank_columns.append(
            _rank_vectors(
                suspect_activations[layer_name],
                suspect_predictions,
                ref_layer_activations,
                reference_predictions,
                radius,
                leave_one_out=False,
            )
        )

    clean_rank_vectors = np.stack(reference_rank_columns, axis=1)
    suspect_rank_vectors = np.stack(suspect_rank_columns, axis=1)
    if np.all(clean_rank_vectors == clean_rank_vectors[0]):
        clean_errors = np.zeros(clean_rank_vectors.shape[0], dtype=np.float64)
        suspect_errors = np.mean(
            np.square(suspect_rank_vectors - clean_rank_vectors[0]), axis=1
        )
    else:
        n_components = min(
            2, clean_rank_vectors.shape[0] - 1, clean_rank_vectors.shape[1]
        )
        if n_components < 1:
            raise ValueError("TED++ needs at least two clean reference rank vectors.")

        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(clean_rank_vectors)

        clean_reconstruction = pca.inverse_transform(
            pca.transform(clean_rank_vectors)
        )
        clean_errors = np.mean(
            np.square(clean_rank_vectors - clean_reconstruction), axis=1
        )
        suspect_reconstruction = pca.inverse_transform(
            pca.transform(suspect_rank_vectors)
        )
        suspect_errors = np.mean(
            np.square(suspect_rank_vectors - suspect_reconstruction), axis=1
        )

    threshold = float(np.quantile(clean_errors, 1.0 - contamination))
    flagged = suspect_errors > threshold
    num_flagged = int(flagged.sum())
    flagged_fraction = num_flagged / len(suspect_errors)

    return {
        "mode": "input-level",
        "method": "tedpp",
        "verdict": (
            "likely backdoored"
            if flagged_fraction > contamination
            else "likely clean"
        ),
        "num_inputs": len(suspect_errors),
        "num_flagged": num_flagged,
        "threshold": threshold,
        "parameters": {
            "reference_size": reference_count,
            "contamination": contamination,
            "batch_size": batch_size,
            "seed": seed,
            "beta": beta,
            "samples_per_class": samples_per_class,
            "tube_radii": layer_radii,
        },
        "per_sample": [
            {"score": float(score), "flagged": bool(is_flagged)}
            for score, is_flagged in zip(suspect_errors, flagged)
        ],
    }


class TEDPLUS:
    """Object-style wrapper for :func:`run_tedpp`."""

    def __init__(
        self,
        model: nn.Module,
        reference_data: Dataset,
        suspect_data: Dataset,
        reference_size: int = 1000,
        contamination: float = 0.01,
        batch_size: int = 100,
        device: str | torch.device = "cpu",
        beta: float = 0.5,
        samples_per_class: int = 5,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.reference_data = reference_data
        self.suspect_data = suspect_data
        self.reference_size = reference_size
        self.contamination = contamination
        self.batch_size = batch_size
        self.device = device
        self.beta = beta
        self.samples_per_class = samples_per_class
        self.seed = seed

    def detect(self) -> dict[str, Any]:
        return run_tedpp(
            model=self.model,
            reference_data=self.reference_data,
            suspect_data=self.suspect_data,
            reference_size=self.reference_size,
            contamination=self.contamination,
            batch_size=self.batch_size,
            seed=self.seed,
            device=self.device,
            beta=self.beta,
            samples_per_class=self.samples_per_class,
        )

    def test(self) -> dict[str, Any]:
        return self.detect()
