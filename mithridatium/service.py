from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from mithridatium import loader
from mithridatium import utils
from mithridatium.defenses.aeva import run_aeva
from mithridatium.defenses.freeeagle import run_freeeagle
from mithridatium.defenses.mmbd import get_device
from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.defenses.strip import strip_scores

DEFENSES = {"freeeagle", "aeva", "mmbd", "strip"}
PROVIDERS = {"torchvision", "huggingface"}


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


def _apply_freeeagle_overrides(config: Any, overrides: Optional[dict[str, Any]]) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "freeeagle_num_classes" and int(value) <= 0:
            continue
        setattr(config, key, value)


def run_detection(
    *,
    model: str = "",
    data: str,
    defense: str,
    provider: str = "torchvision",
    hf_model_id: str = "microsoft/resnet-50",
    device_index: int = 0,
    progress: Optional[Callable[[str], None]] = None,
    freeeagle_options: Optional[dict[str, Any]] = None,
    aeva_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    defense_key = defense.strip().lower()
    if defense_key not in DEFENSES:
        supported = ", ".join(sorted(DEFENSES))
        raise DetectionUsageError(
            f"unsupported defense '{defense}'. Supported defenses: {supported}"
        )

    provider_key = provider.strip().lower()
    if provider_key not in PROVIDERS:
        supported = ", ".join(sorted(PROVIDERS))
        raise DetectionUsageError(
            f"unsupported provider '{provider}'. Supported providers: {supported}"
        )

    dataset_key = data.strip().lower()
    try:
        cfg = utils.get_preprocess_config(dataset_key)
    except Exception as ex:
        raise DetectionUsageError(str(ex)) from ex

    num_classes = cfg.get_num_classes()
    model_ref = ""

    if provider_key == "torchvision":
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
            mdl, _ = loader.detect_and_build(
                str(ckpt_path),
                arch_hint="resnet18",
                num_classes=num_classes,
            )
        except Exception as ex:
            raise DetectionIOError(
                f"failed to load local checkpoint '{ckpt_path}'.\nReason: {ex}"
            ) from ex
    else:
        model_ref = hf_model_id.strip()
        if not model_ref:
            raise DetectionUsageError(
                "provider 'huggingface' requires --hf-model-id (or a non-empty UI model ID)."
            )
        try:
            from mithridatium import loader_hf
        except Exception as ex:
            raise DetectionUsageError(
                "provider 'huggingface' requires optional dependencies. "
                "Install with: pip install -e '.[hf]'"
            ) from ex

        _emit(progress, f"[service] loading Hugging Face model '{model_ref}'...")
        try:
            mdl, _ = loader_hf.build_huggingface_model(model_ref)
        except Exception as ex:
            raise DetectionIOError(
                f"failed to load Hugging Face model '{model_ref}'.\nReason: {ex}"
            ) from ex

        if hasattr(mdl, "get_preprocess_config"):
            try:
                cfg = mdl.get_preprocess_config(fallback_dataset=dataset_key)
            except Exception:
                cfg = utils.get_preprocess_config(dataset_key)

    _emit(progress, "[service] validating model (architecture + dry forward)...")
    try:
        loader.validate_model(mdl, "auto", cfg.get_input_size())
    except Exception as ex:
        raise DetectionIOError(f"model validation failed.\n{ex}") from ex
    _emit(progress, "[service] model validation OK")

    _emit(progress, "[service] building dataloader...")
    try:
        _, config = utils.dataloader_for(dataset_key, "test", 256)
        if provider_key == "huggingface":
            config.set_input_size(tuple(cfg.get_input_size()))
            config.set_mean(tuple(cfg.get_mean()))
            config.set_std(tuple(cfg.get_std()))
            config.set_normalize(bool(cfg.get_normalize()))
            config.set_dataset(dataset_key)
    except Exception as ex:
        raise DetectionIOError(
            f"failed to build dataloader for dataset '{dataset_key}'.\nReason: {ex}"
        ) from ex

    if defense_key == "freeeagle":
        _apply_freeeagle_overrides(config, freeeagle_options)

    _emit(progress, f"[service] running defense={defense_key}...")
    try:
        device = get_device(device_index)
        mdl = mdl.to(device)

        if defense_key == "mmbd":
            results = run_mmbd(mdl, config, device=device)
        elif defense_key == "aeva":
            aeva_kwargs = dict(aeva_options or {})
            results = run_aeva(
                mdl,
                config,
                task=dataset_key,
                device=device,
                model_path=model_ref,
                **aeva_kwargs,
            )
        elif defense_key == "freeeagle":
            results = run_freeeagle(mdl, config, device=device)
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
