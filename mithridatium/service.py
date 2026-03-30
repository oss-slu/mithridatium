from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from mithridatium import loader
from mithridatium import utils
from mithridatium.defenses.aeva import run_aeva
from mithridatium.defenses.mmbd import get_device
from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.defenses.strip import strip_scores

DEFENSES = {"aeva", "mmbd", "strip"}


class DetectionError(RuntimeError):
    """Base class for recoverable detection failures."""


class DetectionUsageError(DetectionError):
    """Invalid or unsupported user input."""


class DetectionNoInputError(DetectionError):
    """Input file was missing or not a regular file."""


class DetectionIOError(DetectionError):
    """Input file or model failed to load/validate."""


class DetectionExecutionError(DetectionError):
    """Defense execution failed after setup."""


def _emit(progress: Optional[Callable[[str], None]], message: str) -> None:
    if progress is not None:
        progress(message)
    else:
        print(message)


def run_detection(
    *,
    model: str,
    data: str,
    defense: str,
    device_index: int = 0,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    defense_key = defense.strip().lower()
    if defense_key not in DEFENSES:
        supported = ", ".join(sorted(DEFENSES))
        raise DetectionUsageError(
            f"unsupported defense '{defense}'. Supported defenses: {supported}"
        )

    dataset_key = data.strip().lower()
    try:
        cfg = utils.get_preprocess_config(dataset_key)
    except Exception as ex:
        raise DetectionUsageError(str(ex)) from ex

    num_classes = cfg.get_num_classes()
    ckpt_path = Path(model)
    if not ckpt_path.exists() or not ckpt_path.is_file():
        raise DetectionNoInputError(f"model path not found or not a file: {ckpt_path}")
    try:
        with ckpt_path.open("rb"):
            pass
    except OSError as ex:
        raise DetectionIOError(
            f"model file could not be opened: {ckpt_path}\nReason: {ex}"
        ) from ex

    if ckpt_path.suffix.lower() not in {".pt", ".pth"}:
        raise DetectionUsageError(
            f"unsupported model extension '{ckpt_path.suffix}'. Expected .pth or .pt"
        )

    model_ref = str(ckpt_path)
    _emit(progress, "[service] loading local model checkpoint...")
    try:
        # `detect_and_build` inspects checkpoint tensors and selects the matching architecture variant.
        mdl, _ = loader.detect_and_build(
            str(ckpt_path),
            arch_hint="resnet18",
            num_classes=num_classes,
        )
    except Exception as ex:
        raise DetectionIOError(
            f"failed to load local checkpoint '{ckpt_path}'.\nReason: {ex}"
        ) from ex

    _emit(progress, "[service] validating model (architecture + dry forward)...")
    try:
        loader.validate_model(mdl, "auto", cfg.get_input_size())
    except Exception as ex:
        raise DetectionIOError(f"model validation failed.\n{ex}") from ex
    _emit(progress, "[service] model validation OK")

    _emit(progress, "[service] building dataloader...")
    try:
        _, config = utils.dataloader_for(dataset_key, "test", 256)
    except Exception as ex:
        raise DetectionIOError(
            f"failed to build dataloader for dataset '{dataset_key}'.\nReason: {ex}"
        ) from ex

    _emit(progress, f"[service] running defense={defense_key}...")
    try:
        device = get_device(device_index)
        mdl = mdl.to(device)

        if defense_key == "mmbd":
            results = run_mmbd(mdl, config, device=device)
        elif defense_key == "aeva":
            results = run_aeva(
                mdl,
                config,
                task=dataset_key,
                device=device,
                model_path=model_ref,
            )
        else:
            results = strip_scores(mdl, config, device=device)
    except Exception as ex:
        raise DetectionExecutionError(
            f"failed to run '{defense_key}' on model {model_ref}.\nReason: {ex}"
        ) from ex

    return {
        "model_ref": model_ref,
        "defense": defense_key,
        "dataset": dataset_key,
        "results": results,
    }
