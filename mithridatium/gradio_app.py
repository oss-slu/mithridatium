from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from typing import Optional

try:
    import gradio as gr
except ImportError as ex:
    raise ImportError(
        "gradio is required for the Mithridatium UI. Install with: pip install -e '.[ui]'"
    ) from ex

from mithridatium import report as rpt
from mithridatium import utils
from mithridatium.constants import VERSION
from mithridatium.service import DEFENSES
from mithridatium.service import run_detection

DATASET_CHOICES = sorted(utils.DATASET_CONFIGS.keys())


def _write_json_file(payload: dict[str, Any], out_path: str) -> str:
    path = Path(out_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path.resolve())


def _write_temp_json(payload: dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix="mithridatium-report-",
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        return tmp.name


def _run_detection_from_ui(
    model_path: str,
    model_file: Optional[str],
    dataset: str,
    defense: str,
    save_report: bool,
    report_out: str,
) -> tuple[str, str, str, dict[str, Any], str | None, str]:
    logs: list[str] = []

    def _capture(msg: str) -> None:
        logs.append(msg)

    try:
        selected_model = (model_file or "").strip() or str(model_path).strip()
        if not selected_model:
            raise ValueError("Provide a local model path or pick a model file.")

        logs.append(f"[ui] model source: {selected_model}")
        detection = run_detection(
            model=selected_model,
            data=dataset,
            defense=defense,
            progress=_capture,
        )

        rep = rpt.build_report(
            model_path=detection["model_ref"],
            defense=detection["defense"],
            dataset=detection["dataset"],
            version=VERSION,
            results=detection["results"],
        )
        summary = rpt.render_summary(rep)
        raw_verdict = str(rep.get("results", {}).get("verdict", "")).strip()
        verdict_lower = raw_verdict.lower()
        if "clean" in verdict_lower:
            verdict = "Not backdoored"
        elif "backdoor" in verdict_lower:
            verdict = "Backdoored"
        else:
            verdict = raw_verdict or "Unknown"

        if save_report:
            target = str(report_out).strip() or "reports/gradio_report.json"
            _write_json_file(rep, target)
            logs.append(f"[ui] saved report: {target}")

        download_file = _write_temp_json(rep)
        status = "Detection complete."
        return status, verdict, summary, rep, download_file, "\n".join(logs)
    except Exception as ex:
        err = f"Error: {ex}"
        return err, "Unknown", err, {"error": str(ex)}, None, "\n".join(logs)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mithridatium UI") as demo:
        gr.Markdown(
            "## Mithridatium Detection UI\n"
            "Load a local checkpoint, pick a defense, and inspect the report. "
            "Model architecture is auto-detected from checkpoint weights."
        )

        defense = gr.Dropdown(
            choices=sorted(DEFENSES),
            value="mmbd",
            label="Defense",
        )

        model_path = gr.Textbox(
            value="models/resnet18.pth",
            label="Local Model Path (.pth or .pt)",
        )
        model_file = gr.File(
            label="Or Pick Model File (.pth or .pt)",
            file_count="single",
            type="filepath",
        )

        dataset = gr.Dropdown(
            choices=DATASET_CHOICES,
            value="cifar10",
            label="Dataset",
        )

        with gr.Accordion("Report Output", open=False):
            save_report = gr.Checkbox(
                value=True,
                label="Persist report to file",
            )
            report_out = gr.Textbox(
                value="reports/gradio_report.json",
                label="Report Path",
            )

        run_btn = gr.Button("Run Detection", variant="primary")

        status = gr.Textbox(label="Status")
        verdict = gr.Textbox(label="Verdict")
        summary = gr.Textbox(label="Summary", lines=10)
        report_json = gr.JSON(label="Report JSON")
        report_file = gr.File(label="Download JSON")
        logs = gr.Textbox(label="Run Log", lines=10)

        run_btn.click(
            fn=_run_detection_from_ui,
            inputs=[model_path, model_file, dataset, defense, save_report, report_out],
            outputs=[status, verdict, summary, report_json, report_file, logs],
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch()
