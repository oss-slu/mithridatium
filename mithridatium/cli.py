# mithridatium/cli.py
import typer
import json
from pathlib import Path
import sys
from mithridatium import report as rpt
from mithridatium import loader as loader
from mithridatium import loader_hf as loader_hf
from mithridatium import utils
from mithridatium.defenses.aeva import run_aeva
from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.defenses.freeeagle import run_freeeagle
from mithridatium.defenses.strip import strip_scores
from mithridatium.defenses.mmbd import get_device
from mithridatium.loader import validate_model
from mithridatium.defenses.aeva import run_aeva



VERSION = "0.1.1"
DEFENSES = {"freeeagle", "aeva", "mmbd", "strip"}

EXIT_USAGE_ERROR = 64     # invalid CLI usage (e.g., unsupported --defense)
EXIT_NO_INPUT = 66        # input file missing/not a file
EXIT_CANT_CREATE = 73     # cannot create/overwrite output without --force
EXIT_IO_ERROR = 74        # input exists but can't be opened/read

app = typer.Typer(help="Mithridatium CLI - verify pretrained model integrity")


def _write_json(obj: dict, out_path: str, force: bool) -> None:
    """
    Write JSON to a file or to stdout.
    - Stdout using "--out -"
    - Overwrite using "--force"

    """

    if out_path == "-":
        json.dump(obj, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Checks if file exists and prevents overwriting. Use --force to override.
    if path.exists() and not force:
        typer.secho(
            f"Error: output file already exists: {path}.",
        )
        raise typer.Exit(code=EXIT_CANT_CREATE)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@app.callback(invoke_without_command=True)
def _root(
    # This is a calback that prints the version whenever it is ran.
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show Mithridatium version and exit.",
        is_eager=True, # ensures this runs before any command (including --help
    )
):

    if version:
        typer.echo(VERSION)
        raise typer.Exit()

@app.command()
def defenses() -> None:
    """
    List supported defenses.
    """
    for d in sorted(DEFENSES):
        typer.echo(d)

@app.command()
def detect(
    model: str = typer.Option(
        "models/resnet18.pth",
        "--model",
        "-m",
        help="The model path .pth. E.g. 'models/resnet18.pth'.",
    ),
    data: str = typer.Option(
        "cifar10",
        "--data",
        "-d",
        help="The dataset name. E.g. 'cifar10'.",
    ),
    defense: str = typer.Option(
        "mmbd",
        "--defense",
        "-D",
        help="The defense you want to run. E.g. 'mmbd', 'strip', 'aeva', or 'freeeagle'.",
    ),
    arch: str = typer.Option(
        "resnet18",
        "--arch",
        "-a",
        help="The model architecture to use. E.g. 'resnet18' or 'hf_resnet50'.",
    ),
    provider: str = typer.Option(
        "torchvision",
        "--provider",
        "-p",
        help="Model provider: 'torchvision' or 'huggingface'.",
    ),
    hf_model_id: str = typer.Option(
        "microsoft/resnet-50",
        "--hf-model-id",
        help="Hugging Face model ID when --provider huggingface is used.",
    ),
    out: str = typer.Option(
        "reports/report.json",
        "--out",
        "-o",
        help='The output path for the JSON report. Use "-" for stdout or a file path (e.g. "reports/report.json").',
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="This allows overwriting. E.g. if the output file already exists --force will overwrite it.",
    ),
    freeeagle_num_classes: int = typer.Option(
        0,
        "--freeeagle-num-classes",
        help="FreeEagle override for number of classes. Use 0 to auto-infer from model head.",
    ),
    freeeagle_num_dummy: int = typer.Option(
        1,
        "--freeeagle-num-dummy",
        help="FreeEagle number of dummy optimization vectors.",
    ),
    freeeagle_num_important_neurons: int = typer.Option(
        5,
        "--freeeagle-num-important-neurons",
        help="FreeEagle top neurons used when computing tendency.",
    ),
    freeeagle_metric: str = typer.Option(
        "softmax_score",
        "--freeeagle-metric",
        help="FreeEagle anomaly metric (e.g. 'softmax_score').",
    ),
    freeeagle_use_transpose_correction: bool = typer.Option(
        False,
        "--freeeagle-use-transpose-correction",
        help="Enable transpose correction inside FreeEagle.",
    ),
    freeeagle_bound_on: bool = typer.Option(
        True,
        "--freeeagle-bound-on/--freeeagle-no-bound-on",
        help="Enable or disable bounded optimization in FreeEagle.",
    ),
    freeeagle_optimize_steps: int = typer.Option(
        300,
        "--freeeagle-optimize-steps",
        help="FreeEagle optimization steps.",
    ),
    freeeagle_learning_rate: float = typer.Option(
        1e-2,
        "--freeeagle-learning-rate",
        help="FreeEagle optimization learning rate.",
    ),
    freeeagle_weight_decay: float = typer.Option(
        5e-3,
        "--freeeagle-weight-decay",
        help="FreeEagle optimization weight decay.",
    ),
    freeeagle_anomaly_threshold: float = typer.Option(
        2.0,
        "--freeeagle-anomaly-threshold",
        help="Threshold for FreeEagle anomaly_metric verdict.",
    ),
    freeeagle_inspect_layer_position: int = typer.Option(
        2,
        "--freeeagle-inspect-layer-position",
        help="ResNet stage index inspected by FreeEagle (0..4).",
    ),
    aeva_samples_per_class: int = typer.Option(
        10,
        "--aeva-samples-per-class",
        help="AEVA number of samples per class.",
    ),
    aeva_hsja_iterations: int = typer.Option(
        10,
        "--aeva-hsja-iterations",
        help="AEVA HSJA iterations.",
    ),
    aeva_hsja_max_num_evals: int = typer.Option(
        2000,
        "--aeva-hsja-max-num-evals",
        help="AEVA HSJA max number of evaluations.",
    ),
    aeva_hsja_init_num_evals: int = typer.Option(
        50,
        "--aeva-hsja-init-num-evals",
        help="AEVA HSJA initial number of evaluations.",
    ),
    aeva_hsja_query_batch_size: int = typer.Option(
        256,
        "--aeva-hsja-query-batch-size",
        help="AEVA HSJA query batch size.",
    ),
    aeva_anomaly_index_threshold: float = typer.Option(
        4.0,
        "--aeva-anomaly-index-threshold",
        help="AEVA anomaly threshold.",
    ),
    aeva_verbose: bool = typer.Option(
        False,
        "--aeva-verbose",
        help="Enable verbose AEVA logging.",
    ),
    aeva_sp: int = typer.Option(
        0,
        "--aeva-sp",
        help="AEVA start source class index.",
),
    aeva_ep: int = typer.Option(
        1,
        "--aeva-ep",
        help="AEVA exclusive end source class index.",
),
):
    """
    Run a supported defense against either a local checkpoint or a Hugging Face model.
    """
    provider = provider.strip().lower()

    if provider not in {"torchvision", "huggingface"}:
        typer.secho(
            f"Error: unsupported --provider '{provider}'. Supported providers: torchvision, huggingface",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    if provider == "torchvision":
        p = Path(model)

        if not p.exists() or not p.is_file():
            typer.secho(
                f"Error: model path not found or not a file: {p}", err=True
            )
            raise typer.Exit(code=EXIT_NO_INPUT)

        try:
            with p.open("rb"):
                pass
        except OSError as ex:
            typer.secho(
                f"Error: model file could not be opened: {p}\nReason: {ex}", err=True
            )
            raise typer.Exit(code=EXIT_IO_ERROR)
    else:
        p = None

    d = defense.strip().lower()
    if d not in DEFENSES:
        typer.secho(
            "Error: unsupported --defense "
            f"'{defense}'. Supported defenses: {', '.join(sorted(DEFENSES))}",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    cfg = utils.get_preprocess_config(data)
    num_classes = cfg.get_num_classes()

    print(f"[cli] loading model from provider={provider}…")

    if provider == "torchvision":
        mdl, feature_module = loader.detect_and_build(str(p), arch_hint=arch, num_classes=10)
        cfg = utils.get_preprocess_config(data)
    else:
        mdl, feature_module = loader.build_huggingface_model(hf_model_id)
        if hasattr(mdl, "get_preprocess_config"):
            cfg = mdl.get_preprocess_config(fallback_dataset=data)
        else:
            cfg = utils.get_preprocess_config(data)

    try:
        print("[cli] validating model (architecture + dry forward)…")
        input_size = cfg.get_input_size()
        loader.validate_model(mdl, arch, input_size)
        print("[cli] model validation OK")
    except Exception as ex:
        typer.secho(
            f"Error: model validation failed.\n{ex}",
            err=True,
        )
        raise typer.Exit(code=EXIT_IO_ERROR)

    print("[cli] building dataloader…")
    if provider == "huggingface":
        test_loader, config = utils.dataloader_for_config(data, "test", cfg, 256)
    else:
        _, config = utils.dataloader_for(data, "test", 256)

    if d == "freeeagle":
        if freeeagle_num_classes > 0:
            setattr(config, "freeeagle_num_classes", freeeagle_num_classes)
        setattr(config, "freeeagle_num_dummy", freeeagle_num_dummy)
        setattr(config, "freeeagle_num_important_neurons", freeeagle_num_important_neurons)
        setattr(config, "freeeagle_metric", freeeagle_metric)
        setattr(config, "freeeagle_use_transpose_correction", freeeagle_use_transpose_correction)
        setattr(config, "freeeagle_bound_on", freeeagle_bound_on)
        setattr(config, "freeeagle_optimize_steps", freeeagle_optimize_steps)
        setattr(config, "freeeagle_learning_rate", freeeagle_learning_rate)
        setattr(config, "freeeagle_weight_decay", freeeagle_weight_decay)
        setattr(config, "freeeagle_anomaly_threshold", freeeagle_anomaly_threshold)
        setattr(config, "freeeagle_inspect_layer_position", freeeagle_inspect_layer_position)
        
    model_ref = str(p) if provider == "torchvision" else hf_model_id

    try:
        loader.ensure_defense_compatibility(mdl, d, feature_module)
    except ValueError as ex:
        typer.secho(
            f"Error: defense/model compatibility check failed.\nReason: {ex}",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    print(f"[cli] running defense={d}…")
    try:
        device = get_device(0)
        mdl = mdl.to(device)

        if d == "mmbd":
            results = run_mmbd(mdl, config)
        elif d == "aeva":
            results = run_aeva(
                mdl,
                config,
                task=data,
                device=device,
                model_path=(str(p) if provider == "torchvision" else hf_model_id),
                sp=aeva_sp,
                ep=aeva_ep,
                samples_per_class=aeva_samples_per_class,
                hsja_iterations=aeva_hsja_iterations,
                hsja_max_num_evals=aeva_hsja_max_num_evals,
                hsja_init_num_evals=aeva_hsja_init_num_evals,
                hsja_query_batch_size=aeva_hsja_query_batch_size,
                anomaly_index_threshold=aeva_anomaly_index_threshold,
                verbose=aeva_verbose,
            )
        elif d == "strip":
            results = strip_scores(mdl, config, test_loader=test_loader)
        elif d == "freeeagle":
            results = run_freeeagle(mdl, config)
        else:
            raise ValueError(f"Unsupported defense '{d}'.")


    except Exception as ex:
        typer.secho(
            f"Error: failed to run '{d}' on model {model_ref}.\nReason: {ex}",
            err=True,
        )
        raise typer.Exit(code=EXIT_IO_ERROR)

    rep = rpt.build_report(
        model_path=model_ref,
        defense=d,
        dataset=data,
        version=VERSION,
        results=results,
    )

    try:
        rpt.validate_report_data(rep)
    except Exception as ex:
        typer.secho(
            f"Error: generated report failed schema validation.\nReason: {ex}",
            err=True,
        )
        raise typer.Exit(code=EXIT_IO_ERROR)

    _write_json(rep, out, force)
    print(rpt.render_summary(rep))

if __name__ == "__main__":
    app()
