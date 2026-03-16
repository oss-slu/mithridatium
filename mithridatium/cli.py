# mithridatium/cli.py
import typer
import json
from pathlib import Path
import sys
from mithridatium import report as rpt
from mithridatium import loader as loader
from mithridatium import utils
from mithridatium.defenses.aeva import run_aeva
from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.defenses.strip import strip_scores
from mithridatium.defenses.mmbd import get_device
from mithridatium.loader import validate_model



VERSION = "0.1.1"
DEFENSES = {"aeva", "mmbd", "strip"}

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


def dummy_report(model_path: str, defense: str, out_path: str, force: bool) -> None:
    """
    Nothing runs yet, just a dummy report.
    """
    
    # dummy report:
    report = {
        "mithridatium_version": VERSION,
        "model_path": model_path,
        "defense": defense,
        "status": "Not yet implemented", 
    }

    _write_json(report, out_path, force)
    where = "stdout" if out_path == "-" else out_path
    typer.echo(f"Report written to {where}")


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
        help="The defense you want to run. E.g. 'aeva', 'mmbd', or 'strip'.",
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

    print(f"[cli] loading model from provider={provider}…")

    if provider == "torchvision":
        mdl, feature_module = loader.detect_and_build(str(p), arch_hint=arch, num_classes=10)
    else:
        mdl, feature_module = loader.build_huggingface_model(hf_model_id)

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
    test_loader, config = utils.dataloader_for(data, "test", 256)

    model_ref = str(p) if provider == "torchvision" else hf_model_id

    print(f"[cli] running defense={d}…")
    try:
        device = get_device(0)
        mdl = mdl.to(device)

        if d == "mmbd":
            results = run_mmbd(mdl, config)
        elif d == "aeva":
            results = run_aeva(mdl, config, task=data, device=device, model_path=p)
        elif d == "strip":
            results = strip_scores(mdl, config)
        else:
            results = {
                "suspected_backdoor": False,
                "num_flagged": 0,
                "top_eigenvalue": 0.0,
            }

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
    _write_json(rep, out, force)
    print(rpt.render_summary(rep))

if __name__ == "__main__":
    app()
