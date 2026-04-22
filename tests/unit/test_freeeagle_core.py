import numpy as np
import torch

from mithridatium.defenses._freeeagle_core import FreeEagleCoreConfig, inspect_model_embeddings


def test_inspect_model_embeddings_returns_expected_shapes():
    torch.manual_seed(0)

    num_classes = 4
    embedding_shape = (1, 8)

    head = torch.nn.Linear(8, num_classes)

    def model_head(x):
        x = x.view(x.size(0), -1)
        return head(x)

    config = FreeEagleCoreConfig(
        num_classes=num_classes,
        num_dummy=1,
        metric="softmax_score",
        optimize_steps=3,
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    outputs = inspect_model_embeddings(
        model_head=model_head,
        embedding_shape=embedding_shape,
        config=config,
        device=torch.device("cpu"),
    )

    assert set(outputs.keys()) == {"anomaly_matrix", "tendency_per_target", "anomaly_metric"}
    assert outputs["anomaly_matrix"].shape == (num_classes, num_classes)
    assert outputs["tendency_per_target"].shape == (num_classes,)
    assert isinstance(outputs["anomaly_metric"], float)


def test_inspect_model_embeddings_anomaly_matrix_non_negative():
    torch.manual_seed(1)

    num_classes = 3
    embedding_shape = (1, 6)
    head = torch.nn.Linear(6, num_classes)

    def model_head(x):
        x = x.view(x.size(0), -1)
        return head(x)

    config = FreeEagleCoreConfig(
        num_classes=num_classes,
        num_dummy=1,
        metric="logit",
        optimize_steps=2,
        learning_rate=1e-2,
        weight_decay=0.0,
    )

    outputs = inspect_model_embeddings(
        model_head=model_head,
        embedding_shape=embedding_shape,
        config=config,
        device=torch.device("cpu"),
    )

    matrix = outputs["anomaly_matrix"]
    assert np.all(matrix >= 0.0)