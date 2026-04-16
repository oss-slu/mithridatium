from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
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
from mithridatium.service import DEFENSES
from mithridatium.service import run_detection

try:
    VERSION = package_version("mithridatium")
except PackageNotFoundError:
    VERSION = "0.1.1"

DATASET_CHOICES = sorted(utils.DATASET_CONFIGS.keys())
PROVIDER_CHOICES = ["torchvision", "huggingface"]
UI_CSS = """
.gradio-container .run-detection-btn,
.gradio-container .run-detection-btn button,
.gradio-container button.run-detection-btn {
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
}

.gradio-container .run-detection-btn:hover,
.gradio-container .run-detection-btn button:hover,
.gradio-container button.run-detection-btn:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}
"""


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
    provider: str,
    model_path: str,
    model_file: Optional[str],
    hf_model_id: str,
    dataset: str,
    defense: str,
    save_report: bool,
    report_out: str,
) -> tuple[str, str, str, dict[str, Any], str | None, str]:
    logs: list[str] = []

    def _capture(msg: str) -> None:
        logs.append(msg)

    try:
        provider_key = str(provider).strip().lower()
        if provider_key not in PROVIDER_CHOICES:
            raise ValueError(f"Unsupported provider '{provider}'.")

        selected_model = ""
        selected_hf_model_id = str(hf_model_id).strip()

        if provider_key == "huggingface":
            if not selected_hf_model_id:
                raise ValueError("Provide a Hugging Face model ID.")
            logs.append(f"[ui] provider: {provider_key}")
            logs.append(f"[ui] model id: {selected_hf_model_id}")
        else:
            selected_model = (model_file or "").strip() or str(model_path).strip()
            if not selected_model:
                raise ValueError("Provide a local model path or pick a model file.")
            logs.append(f"[ui] provider: {provider_key}")
            logs.append(f"[ui] model source: {selected_model}")

        detection = run_detection(
            model=selected_model,
            data=dataset,
            defense=defense,
            provider=provider_key,
            hf_model_id=selected_hf_model_id,
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


def _provider_ui_state(provider: str):
    provider_key = str(provider).strip().lower()
    hf_selected = provider_key == "huggingface"
    return (
        gr.update(visible=not hf_selected),
        gr.update(visible=not hf_selected),
        gr.update(visible=hf_selected),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Mithridatium UI") as demo:
        gr.Markdown(
            "## Mithridatium Detection UI\n"
            "Run a defense against either a local checkpoint or a Hugging Face model ID."
        )

        provider = gr.Dropdown(
            choices=PROVIDER_CHOICES,
            value="torchvision",
            label="Model Provider",
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
        hf_model_id = gr.Textbox(
            value="microsoft/resnet-50",
            label="Hugging Face Model ID",
            visible=False,
        )

        provider.change(
            fn=_provider_ui_state,
            inputs=[provider],
            outputs=[model_path, model_file, hf_model_id],
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

        run_btn = gr.Button(
            "Run Detection",
            variant="primary",
            elem_classes=["run-detection-btn"],
        )

        status = gr.Textbox(label="Status")
        verdict = gr.Textbox(label="Verdict")
        summary = gr.Textbox(label="Summary", lines=10)
        report_json = gr.JSON(label="Report JSON")
        report_file = gr.File(label="Download JSON")
        logs = gr.Textbox(label="Run Log", lines=10)

        run_btn.click(
            fn=_run_detection_from_ui,
            inputs=[
                provider,
                model_path,
                model_file,
                hf_model_id,
                dataset,
                defense,
                save_report,
                report_out,
            ],
            outputs=[status, verdict, summary, report_json, report_file, logs],
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.launch(
        server_name=host,
        server_port=port,
        share=share,
        css=UI_CSS,
    )


if __name__ == "__main__":
    launch()
