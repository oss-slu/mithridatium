from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FreeEagleCoreConfig:
    num_classes: int
    num_dummy: int = 1
    num_important_neurons: int = 5
    metric: str = "softmax_score"
    use_transpose_correction: bool = False
    bound_on: bool = True
    optimize_steps: int = 300
    learning_rate: float = 1e-2
    weight_decay: float = 5e-3
    clamp_min: float = 0.0
    clamp_max: float = 999.0


def optimize_inner_embedding(
    model_head: Callable[[torch.Tensor], torch.Tensor],
    embedding_template: torch.Tensor,
    desired_class: int,
    config: FreeEagleCoreConfig,
    device: torch.device,
) -> torch.Tensor:
    label = torch.tensor([desired_class], device=device)
    dummy_embedding = torch.rand_like(embedding_template, device=device)
    dummy_embedding.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [dummy_embedding],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for _ in range(config.optimize_steps):
        optimizer.zero_grad()
        logits = model_head(dummy_embedding)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        if config.bound_on:
            with torch.no_grad():
                dummy_embedding.clamp_(config.clamp_min, config.clamp_max)

    return dummy_embedding.detach()


def observe_important_neurons_for_class(
    model_head: Callable[[torch.Tensor], torch.Tensor],
    embedding_shape: tuple[int, ...],
    desired_class: int,
    config: FreeEagleCoreConfig,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    embedding_template = torch.ones(size=embedding_shape, device=device)
    dummy_embedding = optimize_inner_embedding(
        model_head=model_head,
        embedding_template=embedding_template,
        desired_class=desired_class,
        config=config,
        device=device,
    )

    sort_obj = torch.sort(dummy_embedding.reshape(-1), descending=True)
    max_indices = sort_obj.indices.detach().cpu().numpy()
    collected_max_indices = max_indices[: config.num_important_neurons]
    return dummy_embedding, collected_max_indices


def compute_dummy_inner_embeddings(
    model_head: Callable[[torch.Tensor], torch.Tensor],
    embedding_shape: tuple[int, ...],
    config: FreeEagleCoreConfig,
    device: torch.device,
) -> list[list[torch.Tensor]]:
    dummies_all: list[list[torch.Tensor]] = [[] for _ in range(config.num_classes)]

    for class_id in range(config.num_classes):
        for _ in range(config.num_dummy):
            dummy_embedding, _ = observe_important_neurons_for_class(
                model_head=model_head,
                embedding_shape=embedding_shape,
                desired_class=class_id,
                config=config,
                device=device,
            )
            dummies_all[class_id].append(dummy_embedding)
    return dummies_all


def compute_metrics_one_source(
    model_head: Callable[[torch.Tensor], torch.Tensor],
    source_class: int,
    dummy_embeddings_all: list[list[torch.Tensor]],
    config: FreeEagleCoreConfig,
) -> np.ndarray:
    dummies_source = dummy_embeddings_all[source_class]
    dummy_avg = torch.zeros_like(dummies_source[0])
    for dummy_embedding in dummies_source:
        dummy_avg += dummy_embedding
    dummy_avg = dummy_avg / config.num_dummy

    logits = model_head(dummy_avg)
    scores = F.softmax(logits, dim=1)

    logits_np = logits.detach().cpu().numpy()[0]
    scores_np = scores.detach().cpu().numpy()[0]
    logits_np[source_class] = 0.0
    scores_np[source_class] = 0.0

    if config.metric == "softmax_score":
        return scores_np
    if config.metric == "logit":
        return logits_np
    raise ValueError(f"Unsupported metric: {config.metric}")


def compute_array_anomaly_metric(values: np.ndarray) -> float:
    flat = values.flatten()
    flat = flat[flat != 0.0]

    if flat.size == 0:
        return 0.0

    flat = np.sort(flat)
    q1 = np.percentile(flat, 25)
    q3 = np.percentile(flat, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 0.0

    return float((np.max(flat) - q3) / iqr)


def inspect_model_embeddings(
    model_head: Callable[[torch.Tensor], torch.Tensor],
    embedding_shape: tuple[int, ...],
    config: FreeEagleCoreConfig,
    device: torch.device,
) -> dict:
    dummy_embeddings_all = compute_dummy_inner_embeddings(
        model_head=model_head,
        embedding_shape=embedding_shape,
        config=config,
        device=device,
    )

    anomaly_matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.float64)

    for source_class in range(config.num_classes):
        metrics = compute_metrics_one_source(
            model_head=model_head,
            source_class=source_class,
            dummy_embeddings_all=dummy_embeddings_all,
            config=config,
        )
        anomaly_matrix[source_class, :] = np.maximum(metrics, 0.0)

    if config.use_transpose_correction:
        eps = 1e-12
        correction = anomaly_matrix / np.maximum(anomaly_matrix.T, eps)
        np.fill_diagonal(correction, 0.0)
        correction = np.where(correction > 1.0, correction.T, correction)
        anomaly_matrix = anomaly_matrix * (1.0 - correction)

    tendency = np.average(anomaly_matrix, axis=0)
    if config.num_classes > 1:
        tendency = tendency * config.num_classes / (config.num_classes - 1)

    anomaly_metric = compute_array_anomaly_metric(tendency)

    return {
        "anomaly_matrix": anomaly_matrix,
        "tendency_per_target": tendency,
        "anomaly_metric": float(anomaly_metric),
    }