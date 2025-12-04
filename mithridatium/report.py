# mithridatium/report.py

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any

def render_summary(report: Dict[str, Any]) -> str:
    r = report["results"]
    return (
        f"Mithridatium {report['mithridatium_version']} | "
        f"defense={report['defense']} | dataset={report['dataset']}\n"
        f"- model_path:        {report['model_path']}\n"
        f"- suspected_backdoor:{r.get('suspected_backdoor')}\n"
        f"- num_flagged:       {r.get('num_flagged')}\n"
        f"- top_eigenvalue:    {r.get('top_eigenvalue')}"
    )

def build_report(
    model_path: str,
    defense: str,
    dataset: str,
    version: str = "0.1.0",
    results: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Single source of truth for a report payload."""
    return {
        "mithridatium_version": version,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "model_path": model_path,
        "defense": defense,
        "dataset": dataset,
        "results": results or {
            # legacy/spectral fallback
            "suspected_backdoor": False,
            "num_flagged": 0,
            "top_eigenvalue": 0.0,
        },
    }

# def mmbd_defense(model, preprocess_config) -> Dict[str, Any]:
#     return run_mmbd(model, preprocess_config)

def render_summary(report: Dict[str, Any]) -> str:
    """Pretty summary that supports both MMBD and legacy outputs."""
    r = report.get("results", {})
    head = (
        f"Mithridatium {report.get('mithridatium_version')} | "
        f"defense={report.get('defense')} | dataset={report.get('dataset')}\n"
        f"- model_path:        {report.get('model_path')}\n"
    )

    defense = report.get("defense")

    # Prefer MMBD-style fields when present
    if defense == "mmbd":
        lines = [head]
        verdict = r.get("verdict")
        if verdict is not None:
            lines.append(f"- verdict:           {verdict}\n")
        pv = r.get("p_value")
        if isinstance(pv, (int, float)):
            lines.append(f"- p_value:           {pv:.6f}\n")
        target = r.get("suspected_target")
        if target is not None:
            lines.append(f"- suspected_target:  {target}\n")
        pcs = r.get("per_class_scores")
        if isinstance(pcs, list):
            lines.append(f"- per_class_scores:  {len(pcs)} classes\n")
        tev = r.get("top_eigenvalue")
        if isinstance(tev, (int, float)):
            lines.append(f"- top_eigenvalue:    {tev}\n")
        return "".join(lines).rstrip()

    if defense == "strip":
        #STRIP Report
        lines = [head]

        # Verdict
        verdict1 = r.get("verdict")
        if verdict1 is not None:
            lines.append(f"- verdict:           {verdict1}\n")

        # Thresholds
        thr = r.get("thresholds", {}).get("entropy_mean_threshold")
        if thr is not None:
            lines.append(f"- entropy_thr:       {thr}\n")

        # Parameters
        params = r.get("parameters", {})
        lines.append(f"- num_bases:         {params.get('num_bases')}\n")
        lines.append(f"- num_perturbations: {params.get('num_perturbations')}\n")

        # Statistics
        stats = r.get("statistics", {})
        lines.append(f"- entropy_mean:      {stats.get('entropy_mean')}\n")
        lines.append(f"- entropy_min:       {stats.get('entropy_min')}\n")
        lines.append(f"- entropy_max:       {stats.get('entropy_max')}\n")

        # Dataset
        ds = r.get("dataset")
        lines.append(f"- dataset:           {ds}\n")

        # Raw entropies
        ent = r.get("entropies")
        if ent:
            lines.append(f"- entropies:\n")
            for idx, e in enumerate(ent):
                lines.append(f"  #{idx}: {e}\n")

        return "".join(lines).rstrip()
    
    # Fallback for legacy/ reports
    return (
        head
        + f"- suspected_backdoor:{r.get('suspected_backdoor')}\n"
        + f"- num_flagged:       {r.get('num_flagged')}\n"
        + f"- top_eigenvalue:    {r.get('top_eigenvalue')}"
    )

def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "reports" / "report_schema.json"

def validate_report_data(data: dict, schema: str | None = None) -> None:
    """
    Validate an in-memory report dict against the JSON Schema.
    Silent on success. Raises on invalid or if jsonschema is missing.
    """
    import json
    from pathlib import Path
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError("jsonschema is required. Install with: pip install jsonschema")

    sch_path = Path(schema) if schema else _schema_path()
    sch = json.loads(sch_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=sch)