import json
import sys
from pathlib import Path

import typer

from mithridatium import report as rpt
from mithridatium.constants import VERSION
from mithridatium.service import DEFENSES
from mithridatium.service import DetectionExecutionError
from mithridatium.service import DetectionIOError
from mithridatium.service import DetectionNoInputError
from mithridatium.service import DetectionUsageError
from mithridatium.service import run_detection

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
        help="The model path (.pth or .pt). E.g. 'models/resnet18.pth'.",
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
    Run a supported defense against a local model checkpoint.
    """
    try:
        detection = run_detection(
            model=model,
            data=data,
            defense=defense,
            progress=print,
        )
    except DetectionUsageError as ex:
        typer.secho(f"Error: {ex}", err=True)
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    except DetectionNoInputError as ex:
        typer.secho(f"Error: {ex}", err=True)
        raise typer.Exit(code=EXIT_NO_INPUT)
    except (DetectionIOError, DetectionExecutionError) as ex:
        typer.secho(f"Error: {ex}", err=True)
        raise typer.Exit(code=EXIT_IO_ERROR)

    rep = rpt.build_report(
        model_path=detection["model_ref"],
        defense=detection["defense"],
        dataset=detection["dataset"],
        version=VERSION,
        results=detection["results"],
    )
    _write_json(rep, out, force)
    print(rpt.render_summary(rep))


@app.command()
def ui(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Interface host (use 0.0.0.0 to expose on local network).",
    ),
    port: int = typer.Option(
        7860,
        "--port",
        help="Port for the Gradio server.",
    ),
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a public Gradio share URL.",
    ),
):
    """
    Launch the Mithridatium Gradio interface.
    """
    try:
        from mithridatium.gradio_app import build_app
    except ImportError:
        typer.secho(
            "Error: Gradio UI requires optional dependency 'gradio'. "
            "Install with: pip install -e '.[ui]'",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)

    demo = build_app()
    demo.launch(server_name=host, server_port=port, share=share)

if __name__ == "__main__":
    app()
