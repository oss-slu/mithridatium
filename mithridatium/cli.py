# mithridatium/cli.py
import typer
import json
from pathlib import Path
import sys
from mithridatium import report as rpt
from mithridatium import loader as loader
from mithridatium import utils
from mithridatium.defenses.mmbd import run_mmbd
from mithridatium.defenses.strip import strip_scores
from mithridatium.defenses.mmbd import get_device
from mithridatium.loader import validate_model



VERSION = "0.1.1"
DEFENSES = {"mmbd", "strip"}

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
        help="The defense you want to run. E.g. 'mmbd' or 'strip'.",
    ),
    arch: str = typer.Option(
        "resnet18",
        "--arch",
        "-a",
        help="The model architecture to use. E.g. 'resnet18'.",
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
    Argument validation:
    1) Model path exists and is a file
    2) File exists but can't be loaded
    3) Unsupported defense
    4) Write dummy JSON (stdout allowed via --out -)
    """
    # 1) Model path exists and is a file
    p = Path(model)
    if not p.exists() or not p.is_file():
        typer.secho(
            f"Error: model path not found or not a file: {p}", err=True
        )
        raise typer.Exit(code=EXIT_NO_INPUT)

    # 2) File exists but can't be loaded
    try:
        with p.open("rb"):
            pass
    except OSError as ex:
        typer.secho(
            f"Error: model file could not be opened: {p}\nReason: {ex}", err=True
        )
        raise typer.Exit(code=EXIT_IO_ERROR)
    
    # 3) Unsupported defense
    d = defense.strip().lower()
    if d not in DEFENSES:
        typer.secho(
            "Error: unsupported --defense "
            f"'{defense}'. Supported defenses: {', '.join(sorted(DEFENSES))}", err=True
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    
        # 4) Build model arch
    print(f"[cli] building model architecture '{arch}'…")
    mdl, feature_module = loader.build_model(arch, num_classes=10)

    # 5) Load weights from checkpoint
    print("[cli] loading weights…")
    mdl = loader.load_weights(mdl, str(p))

    # 6) Validate model BEFORE any defense runs
    # cfg = utils.load_preprocess_config(str(p))  # has input_size etc.
    cfg = utils.get_preprocess_config(data)  # has input_size etc.

    try:
        print("[cli] validating model (architecture + dry forward)…")
        input_size = cfg.get_input_size() 
        validate_model(mdl, arch, input_size)
        print("[cli] model validation OK")
    except Exception as ex:
        typer.secho(
            f"Error: model validation failed.\n{ex}",
            err=True,
        )
        raise typer.Exit(code=EXIT_IO_ERROR)

    # 7) Build dataloader (TEMP: CIFAR-10; replace with PreprocessConfig)
    print("[cli] building dataloader…")
    test_loader, config = utils.dataloader_for(data, "test", 256)


    # 8) Run the defenses that are supported
    print(f"[cli] running defense={d}…")
    try:
        device = get_device(0)
        mdl = mdl.to(device)
        if d == "mmbd":
            # Move model to appropriate device for MMBD
            results = run_mmbd(mdl, config)
        elif d == "strip":
            results = strip_scores(mdl, test_loader, config, device=device)
        else:
            results = {"suspected_backdoor": False, "num_flagged": 0, "top_eigenvalue": 0.0}

    except Exception as ex:
        typer.secho(
            f"Error: failed to run '{d}' on model {p}.\nReason: {ex}", err=True
        )
        raise typer.Exit(code=EXIT_IO_ERROR)
   

    # 8) Build & write report
    rep = rpt.build_report(model_path=str(p), defense=d, dataset=data, version=VERSION, results=results)
    _write_json(rep, out, force)
    print(rpt.render_summary(rep))


if __name__ == "__main__":
    app()