from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from mithridatium.defenses._freeeagle_core import FreeEagleCoreConfig, inspect_model_embeddings
from mithridatium.defenses.mmbd import get_device


def _as_int(configs: Any, key: str, default: int) -> int:
    value = getattr(configs, key, default)
    return int(value)


def _as_float(configs: Any, key: str, default: float) -> float:
    value = getattr(configs, key, default)
    return float(value)


def _as_bool(configs: Any, key: str, default: bool) -> bool:
    value = getattr(configs, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_str(configs: Any, key: str, default: str) -> str:
    value = getattr(configs, key, default)
    return str(value)


class _ResNetAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.stages = [
            nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]

    def embedding_shape(
        self,
        input_shape: tuple[int, int, int],
        inspect_layer_position: int,
        device: torch.device,
    ) -> tuple[int, ...]:
        with torch.no_grad():
            x = torch.zeros((1, *input_shape), device=device)
            emb = self.forward_to_layer(x, inspect_layer_position)
        return tuple(emb.shape)

    def forward_to_layer(self, x: torch.Tensor, inspect_layer_position: int) -> torch.Tensor:
        out = x
        for stage_idx in range(inspect_layer_position + 1):
            out = self.stages[stage_idx](out)
        return out

    def forward_from_layer(self, embedding: torch.Tensor, inspect_layer_position: int) -> torch.Tensor:
        out = embedding
        for stage_idx in range(inspect_layer_position + 1, len(self.stages)):
            out = self.stages[stage_idx](out)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.fc(out)
        return out


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _infer_num_classes(model: nn.Module, fallback: int) -> int:
    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        return int(model.fc.out_features)
    return int(fallback)


def _resolve_input_shape(configs: Any) -> tuple[int, int, int]:
    if hasattr(configs, "get_input_size"):
        shape = tuple(configs.get_input_size())
    else:
        shape = tuple(getattr(configs, "input_size", (3, 32, 32)))

    if len(shape) != 3:
        raise ValueError(f"FreeEagle expects input shape (C,H,W), got {shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _resolve_inspect_layer_position(model: nn.Module, configs: Any) -> int:
    inspect_layer_position = _as_int(configs, "freeeagle_inspect_layer_position", 2)
    max_layer_position = 4
    if inspect_layer_position < 0 or inspect_layer_position > max_layer_position:
        raise ValueError(
            f"freeeagle_inspect_layer_position must be in [0, {max_layer_position}], got {inspect_layer_position}"
        )
    return inspect_layer_position


def run_freeeagle(model, configs, device=None) -> dict:
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device(0)

    model = model.to(device=device, dtype=torch.float32).eval()
    base_model = _unwrap_model(model)

    if base_model.__class__.__name__ != "ResNet":
        raise NotImplementedError(
            "FreeEagle currently supports known ResNet-family models in Mithridatium."
        )

    input_shape = _resolve_input_shape(configs)
    inspect_layer_position = _resolve_inspect_layer_position(base_model, configs)

    default_num_classes = _infer_num_classes(base_model, 10)
    num_classes = _as_int(configs, "freeeagle_num_classes", default_num_classes)

    core_config = FreeEagleCoreConfig(
        num_classes=num_classes,
        num_dummy=_as_int(configs, "freeeagle_num_dummy", 1),
        num_important_neurons=_as_int(configs, "freeeagle_num_important_neurons", 5),
        metric=_as_str(configs, "freeeagle_metric", "softmax_score"),
        use_transpose_correction=_as_bool(configs, "freeeagle_use_transpose_correction", False),
        bound_on=_as_bool(configs, "freeeagle_bound_on", True),
        optimize_steps=_as_int(configs, "freeeagle_optimize_steps", 300),
        learning_rate=_as_float(configs, "freeeagle_learning_rate", 1e-2),
        weight_decay=_as_float(configs, "freeeagle_weight_decay", 5e-3),
    )

    adapter = _ResNetAdapter(base_model)
    embedding_shape = adapter.embedding_shape(
        input_shape=input_shape,
        inspect_layer_position=inspect_layer_position,
        device=device,
    )

    def model_head(embedding: torch.Tensor) -> torch.Tensor:
        return adapter.forward_from_layer(embedding, inspect_layer_position)

    outputs = inspect_model_embeddings(
        model_head=model_head,
        embedding_shape=embedding_shape,
        config=core_config,
        device=device,
    )

    anomaly_metric = float(outputs["anomaly_metric"])
    threshold = _as_float(configs, "freeeagle_anomaly_threshold", 2.0)
    verdict = "likely backdoored" if anomaly_metric >= threshold else "likely clean"

    dataset = configs.get_dataset() if hasattr(configs, "get_dataset") else "unknown"

    return {
        "defense": "freeeagle",
        "anomaly_metric": anomaly_metric,
        "anomaly_matrix": outputs["anomaly_matrix"].tolist(),
        "tendency_per_target": outputs["tendency_per_target"].tolist(),
        "verdict": verdict,
        "thresholds": {
            "anomaly_metric_threshold": threshold,
        },
        "parameters": OrderedDict(
            {
                "num_classes": core_config.num_classes,
                "inspect_layer_position": inspect_layer_position,
                "num_dummy": core_config.num_dummy,
                "num_important_neurons": core_config.num_important_neurons,
                "metric": core_config.metric,
                "use_transpose_correction": core_config.use_transpose_correction,
                "bound_on": core_config.bound_on,
                "optimize_steps": core_config.optimize_steps,
                "learning_rate": core_config.learning_rate,
                "weight_decay": core_config.weight_decay,
                "input_shape": list(input_shape),
            }
        ),
        "dataset": dataset,
    }