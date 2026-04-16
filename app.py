"""
Mithridatium — Streamlit UI
Run: streamlit run app.py
"""

import json
import time
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Mithridatium",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1f1f1f;
}
section[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] p {
    color: #888 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f7f7f5;
    border: 1px solid #e8e8e4;
    border-radius: 6px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #888 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
}

/* Full-width verdict banner */
.banner-clean {
    background: #1b5e20;
    color: #fff;
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 24px;
    font-family: 'IBM Plex Mono', monospace;
}
.banner-bad {
    background: #b71c1c;
    color: #fff;
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 24px;
    font-family: 'IBM Plex Mono', monospace;
}
.banner-unknown {
    background: #212121;
    color: #fff;
    padding: 28px 32px;
    border-radius: 8px;
    margin-bottom: 24px;
    font-family: 'IBM Plex Mono', monospace;
}
.banner-verdict {
    font-size: 28px;
    font-weight: 500;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.banner-sub {
    font-size: 12px;
    opacity: 0.75;
    letter-spacing: 0.08em;
}

/* Verdict badges (inline) */
.verdict-clean {
    background: #e8f5e9; color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 4px; padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500;
    display: inline-block;
}
.verdict-bad {
    background: #fdecea; color: #c62828;
    border: 1px solid #ef9a9a;
    border-radius: 4px; padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500;
    display: inline-block;
}
.verdict-unknown {
    background: #f5f5f5; color: #555;
    border: 1px solid #ddd;
    border-radius: 4px; padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500;
    display: inline-block;
}

/* Section header */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #e8e8e4;
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* Defense card */
.defense-card {
    background: #fff;
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 12px;
}

/* Run button */
.stButton > button {
    background: #0d0d0d !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #222 !important;
}

/* Mono text */
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #555; }

/* Status log */
.status-log {
    background: #0d0d0d;
    color: #4ade80;
    border-radius: 6px;
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    line-height: 1.8;
    max-height: 200px;
    overflow-y: auto;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
DEFENSES = ["mmbd", "strip", "aeva", "freeeagle"]
DATASETS = ["cifar10", "cifar10_for_imagenet", "cifar100", "fake_imagenet"]
PROVIDERS = ["torchvision", "huggingface"]

COLORS = {
    "clean":      "#2e7d32",
    "backdoored": "#c62828",
    "mild":       "#e65100",
    "suspicious": "#b71c1c",
    "normal":     "#2e7d32",
    "neutral":    "#1565c0",
    "bar_clean":  "rgba(46,125,50,0.7)",
    "bar_bad":    "rgba(198,40,40,0.75)",
    "bar_mild":   "rgba(230,81,0,0.75)",
    "bar_normal": "rgba(46,125,50,0.7)",
}

_ALLOWED_EXTENSIONS = {".pth", ".pt"}

def validate_checkpoint_path(raw: str) -> Path:
    """
    Resolve a user-supplied checkpoint path and guard against path traversal.

    Rules:
    - Input must be a non-empty path. Relative paths are allowed,
      and absolute paths are permitted only when they resolve under the
      current working directory or the system temp directory.
    - No ".." traversal is allowed.
    - Extension must be .pth or .pt.

    Raises ValueError on any violation.
    """
    import tempfile as _tempfile

    candidate = Path((raw or "").strip())

    if not str(candidate):
        raise ValueError("Model path cannot be empty.")
    if any(part == ".." for part in candidate.parts):
        raise ValueError("Path traversal is not allowed.")

    p = candidate.resolve(strict=False)
    allowed_bases = (Path.cwd().resolve(), Path(_tempfile.gettempdir().strip()).resolve())

    if not any(p.is_relative_to(base) for base in allowed_bases):
        raise ValueError(
            f"Model path '{p}' is outside allowed directories. "
            "Paths must be relative to the working directory or temp folder."
        )
    if p.suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{p.suffix}'. Expected one of {_ALLOWED_EXTENSIONS}."
        )
    if not p.exists() or not p.is_file():
        raise ValueError(f"Checkpoint not found: '{p}'")
    return p


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=8, r=8, t=8, b=8),
    font=dict(family="IBM Plex Mono, monospace", size=11, color="#555"),
    xaxis=dict(showgrid=False, showline=False, zeroline=False),
    yaxis=dict(gridcolor="#f0f0ec", showline=False, zeroline=False),
)

# ── Helpers ────────────────────────────────────────────────────────────────
def verdict_badge(verdict: str) -> str:
    v = (verdict or "").lower()
    if "clean" in v:
        return f'<span class="verdict-clean">{verdict}</span>'
    elif "backdoor" in v:
        return f'<span class="verdict-bad">{verdict}</span>'
    return f'<span class="verdict-unknown">? {verdict}</span>'



def verdict_banner(verdict: str, model: str, dataset: str, n_flagged: int, n_total: int) -> str:
    v = (verdict or "").lower()
    if "clean" in v:
        cls = "banner-clean"
        label = "LIKELY CLEAN"
    elif "backdoor" in v:
        cls = "banner-bad"
        label = "LIKELY BACKDOORED"
    else:
        cls = "banner-unknown"
        label = "INCONCLUSIVE"
    sub = f"{model} &nbsp;&middot;&nbsp; {dataset} &nbsp;&middot;&nbsp; {n_flagged}/{n_total} defenses flagged"
    return f'''
    <div class="{cls}">
        <div class="banner-verdict">{label}</div>
        <div class="banner-sub">{sub}</div>
    </div>'''

def is_clean(verdict: str) -> bool:
    return "clean" in (verdict or "").lower()


def _to_display_image(image_chw, config) -> np.ndarray:
    import torch

    tensor = image_chw.detach().to(torch.float32).cpu().clone()
    if tensor.ndim != 3:
        raise ValueError("Expected CHW image tensor.")

    if getattr(config, "normalize", True):
        mean = torch.tensor(getattr(config, "mean", (0.0, 0.0, 0.0)), dtype=tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(getattr(config, "std", (1.0, 1.0, 1.0)), dtype=tensor.dtype).view(-1, 1, 1)
        tensor = tensor * std + mean

    low, high = getattr(config, "value_range", (0.0, 1.0))
    tensor = tensor.clamp(float(low), float(high))

    if tensor.shape[0] == 1:
        array = (tensor[0].numpy() * 255.0).astype(np.uint8)
        return array

    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255.0).clip(0, 255).astype(np.uint8)
    return array


def _apply_corner_trigger(image_batch, config, ratio: float = 0.16):
    import torch

    x = image_batch.detach().clone()
    if x.ndim != 4:
        raise ValueError("Expected NCHW tensor batch.")

    _, c, h, w = x.shape
    patch = max(2, int(min(h, w) * ratio))
    y0, x0 = h - patch, w - patch

    if c == 1:
        target_rgb = [1.0]
    else:
        target_rgb = [1.0, 0.0, 0.0] + [0.0] * max(0, c - 3)

    if getattr(config, "normalize", True):
        mean = list(getattr(config, "mean", (0.0, 0.0, 0.0)))
        std = list(getattr(config, "std", (1.0, 1.0, 1.0)))
        channel_values = [
            (target_rgb[i] - mean[i]) / std[i] if i < len(mean) and i < len(std) and std[i] != 0 else target_rgb[i]
            for i in range(c)
        ]
    else:
        channel_values = target_rgb[:c]

    for ch in range(c):
        x[:, ch, y0:h, x0:w] = float(channel_values[ch])
    return x, {"y0": y0, "x0": x0, "size": patch}


def build_trigger_impact_demo(model, test_loader, config, device) -> Optional[dict]:
    import torch

    try:
        batch = next(iter(test_loader))
    except Exception:
        return None

    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        return None

    images, labels = batch[0], batch[1]
    if images is None or len(images) == 0:
        return None

    max_candidates = min(int(images.shape[0]), 32)
    if max_candidates <= 0:
        return None

    perm = torch.randperm(int(images.shape[0]))[:max_candidates]
    candidate_images = images[perm].to(device)
    candidate_labels = labels[perm]

    def _extract_logits(output):
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (list, tuple)) and len(output) > 0:
            return output[0]
        return output

    model.eval()
    with torch.no_grad():
        clean_logits = _extract_logits(model(candidate_images))
        clean_probs = torch.softmax(clean_logits, dim=1)
        clean_preds = clean_probs.argmax(dim=1)

        triggered_batch, patch = _apply_corner_trigger(candidate_images, config)
        triggered_logits = _extract_logits(model(triggered_batch))
        triggered_probs = torch.softmax(triggered_logits, dim=1)
        triggered_preds = triggered_probs.argmax(dim=1)

    true_labels = candidate_labels.to(clean_preds.device).long()
    clean_correct = clean_preds.eq(true_labels)
    triggered_wrong = triggered_preds.ne(true_labels)
    changed = triggered_preds.ne(clean_preds)

    preferred = torch.where(clean_correct & triggered_wrong)[0]
    if preferred.numel() == 0:
        preferred = torch.where(changed)[0]
    if preferred.numel() == 0:
        preferred = torch.where(triggered_wrong)[0]
    if preferred.numel() == 0:
        preferred = torch.tensor([0], device=clean_preds.device)

    chosen_idx = int(preferred[0].item())

    changed_targets = triggered_preds[changed]
    if changed_targets.numel() > 0:
        target_counts = torch.bincount(changed_targets, minlength=clean_probs.shape[1])
        target_class = int(torch.argmax(target_counts).item())
        target_votes = int(target_counts[target_class].item())
        target_total = int(changed_targets.numel())
    else:
        all_counts = torch.bincount(triggered_preds, minlength=clean_probs.shape[1])
        target_class = int(torch.argmax(all_counts).item())
        target_votes = int(all_counts[target_class].item())
        target_total = int(triggered_preds.numel())

    sample = candidate_images[chosen_idx:chosen_idx + 1]
    triggered_sample, _ = _apply_corner_trigger(sample, config)

    true_label = int(true_labels[chosen_idx].item())
    clean_pred = int(clean_preds[chosen_idx].item())
    triggered_pred = int(triggered_preds[chosen_idx].item())
    clean_conf = float(clean_probs[chosen_idx, clean_pred].item())
    triggered_conf = float(triggered_probs[chosen_idx, triggered_pred].item())

    return {
        "true_label": true_label,
        "clean_pred": clean_pred,
        "triggered_pred": triggered_pred,
        "clean_conf": clean_conf,
        "triggered_conf": triggered_conf,
        "clean_correct": bool(clean_pred == true_label),
        "changed_prediction": bool(triggered_pred != clean_pred),
        "misclassified_after_trigger": bool(triggered_pred != true_label),
        "estimated_target_class": target_class,
        "estimated_target_votes": target_votes,
        "estimated_target_total": target_total,
        "num_candidates_evaluated": int(max_candidates),
        "trigger_patch": patch,
        "clean_image": _to_display_image(sample[0], config),
        "triggered_image": _to_display_image(triggered_sample[0], config),
    }


def render_trigger_impact_demo(rep: dict):
    st.markdown("**Trigger impact demo** — same sample before and after a synthetic corner trigger")
    demo = rep.get("_ui_trigger_demo")
    if not demo:
        st.markdown('<p class="mono">trigger demo unavailable for this report (run detection in this session to generate it)</p>', unsafe_allow_html=True)
        return

    dataset_name = (rep.get("dataset") or rep.get("results", {}).get("dataset") or "").lower()
    cifar10_labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    def _label_text(idx) -> str:
        try:
            idx_int = int(idx)
        except Exception:
            return str(idx)

        if dataset_name in {"cifar10", "cifar10_for_imagenet"} and 0 <= idx_int < len(cifar10_labels):
            return f"{idx_int} ({cifar10_labels[idx_int]})"
        return str(idx_int)

    c1, c2 = st.columns(2)
    with c1:
        st.image(
            demo.get("clean_image"),
            caption=(
                f"clean sample · true {_label_text(demo.get('true_label'))} · "
                f"pred {_label_text(demo.get('clean_pred'))} ({demo.get('clean_conf', 0.0):.1%})"
            ),
            use_container_width=True,
        )
    with c2:
        st.image(
            demo.get("triggered_image"),
            caption=(
                f"triggered sample · pred {_label_text(demo.get('triggered_pred'))} "
                f"({demo.get('triggered_conf', 0.0):.1%})"
            ),
            use_container_width=True,
        )

    estimated_target = demo.get("estimated_target_class")
    if estimated_target is not None:
        votes = int(demo.get("estimated_target_votes", 0))
        total = int(demo.get("estimated_target_total", 0))
        st.markdown(
            f'<p class="mono">estimated trigger target class: {_label_text(estimated_target)} '
            f'({votes}/{total} changed samples in this run)</p>',
            unsafe_allow_html=True,
        )

    clean_correct = bool(demo.get("clean_correct", False))
    changed_prediction = bool(demo.get("changed_prediction", False))
    misclassified_after = bool(demo.get("misclassified_after_trigger", False))

    if clean_correct and misclassified_after:
        st.error("Clean sample was correct, and trigger caused a misclassification.", icon=None)
    elif (not clean_correct) and changed_prediction and misclassified_after:
        st.warning("Clean sample was already misclassified; trigger changed it to a different wrong class.", icon=None)
    elif (not clean_correct) and (not changed_prediction):
        st.info("Clean sample was already misclassified, and trigger did not change the predicted class.", icon=None)
    elif changed_prediction:
        st.info("Trigger changed the predicted class, but this sample was not misclassified after trigger.", icon=None)
    else:
        st.info("Trigger did not change this sample’s predicted class.", icon=None)

    patch = demo.get("trigger_patch") or {}
    if patch:
        st.markdown(
            f'<p class="mono">trigger patch: bottom-right square, size={patch.get("size", "?")} px, '
            f'x0={patch.get("x0", "?")}, y0={patch.get("y0", "?")}</p>',
            unsafe_allow_html=True,
        )

def run_detection_cached(model: str, data: str, defense: str, display_name: str = None) -> dict:
    """Run a single defense and return the report dict."""
    import platform
    from mithridatium import report as rpt
    from mithridatium import utils
    from mithridatium import loader
    from mithridatium.defenses.mmbd import get_device

    # Windows multiprocessing workaround — num_workers must be 0
    _is_windows = platform.system() == "Windows"
    if _is_windows:
        import torch.utils.data
        _orig_dataloader = torch.utils.data.DataLoader
        def _safe_dataloader(*args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            return _orig_dataloader(*args, **kwargs)
        torch.utils.data.DataLoader = _safe_dataloader
    VERSION = "0.1.1"

    cfg = utils.get_preprocess_config(data)
    num_classes = cfg.get_num_classes()

    # Determine whether model is a local path or a HuggingFace model ID.
    # A value is treated as local only if it passes full path validation.
    _model_str = (model or "").strip()
    _suffix = Path(_model_str).suffix.lower() if _model_str else ""
    _looks_local = _suffix in _ALLOWED_EXTENSIONS

    is_local = False
    p = None
    if _looks_local:
        try:
            p = validate_checkpoint_path(model)
            is_local = True
        except ValueError as _ve:
            raise ValueError(f"Invalid model path: {_ve}") from _ve

    if is_local:
        # ── Local checkpoint path ────────────────────────────────────────────
        mdl, feature_module = loader.detect_and_build(
            str(p), arch_hint="resnet18", num_classes=num_classes
        )
        model_ref = str(p)
        test_loader, config = utils.dataloader_for(data, "test", 256)
    else:
        # ── HuggingFace model ID ──────────────────────────────────────────
        from mithridatium import loader_hf
        mdl, feature_module = loader_hf.build_huggingface_model(model)
        if hasattr(mdl, "get_preprocess_config"):
            cfg = mdl.get_preprocess_config(fallback_dataset=data)
        test_loader, config = utils.dataloader_for_config(data, "test", cfg, 256)
        model_ref = model

    device = get_device(0)
    mdl = mdl.to(device)

    trigger_demo = None
    try:
        trigger_demo = build_trigger_impact_demo(mdl, test_loader, config, device)
    except Exception:
        trigger_demo = None

    if defense == "mmbd":
        from mithridatium.defenses.mmbd import run_mmbd
        results = run_mmbd(mdl, config, device=device)
    elif defense == "strip":
        from mithridatium.defenses.strip import strip_scores
        results = strip_scores(mdl, config, device=device, test_loader=test_loader)
    elif defense == "aeva":
        from mithridatium.defenses.aeva import run_aeva
        results = run_aeva(mdl, config, task=data, device=device, model_path=model_ref)
    elif defense == "freeeagle":
        from mithridatium.defenses.freeeagle import run_freeeagle
        results = run_freeeagle(mdl, config, device=device)
    else:
        results = {}

    rep = rpt.build_report(
        model_path=display_name if display_name else model_ref,
        defense=defense,
        dataset=data,
        version=VERSION,
        results=results,
    )
    if trigger_demo:
        rep["_ui_trigger_demo"] = trigger_demo
    return rep


# ── Chart builders ─────────────────────────────────────────────────────────
def chart_strip(results: dict) -> go.Figure:
    entropies = results.get("entropies", [])
    threshold = results.get("thresholds", {}).get("entropy_mean_threshold", 0.45)
    colors = [COLORS["bar_bad"] if e < threshold else COLORS["bar_clean"] for e in entropies]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(len(entropies))),
        y=entropies,
        marker_color=colors,
        hovertemplate="sample %{x}<br>entropy: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#c62828",
        line_width=1.5,
        annotation_text=f"threshold {threshold}",
        annotation_font_size=10,
        annotation_font_color="#c62828",
    )
    fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k!="yaxis"}, height=220,
                      yaxis=dict(gridcolor="#f0f0ec", showline=False, zeroline=False, title="entropy"))
    return fig


def chart_strip_dist(results: dict) -> go.Figure:
    entropies = results.get("entropies", [])
    threshold = results.get("thresholds", {}).get("entropy_mean_threshold", 0.45)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=entropies,
        nbinsx=14,
        marker_color=COLORS["neutral"],
        opacity=0.75,
        hovertemplate="entropy: %{x:.2f}<br>count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="#c62828", line_width=1.5)
    fig.add_vline(
        x=results.get("statistics", {}).get("entropy_mean", 0),
        line_dash="dot", line_color="#555", line_width=1,
        annotation_text="mean", annotation_font_size=10,
    )
    fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")}, height=180,
                      xaxis=dict(showgrid=False, showline=False, zeroline=False, title="entropy"),
                      yaxis=dict(gridcolor="#f0f0ec", showline=False, zeroline=False, title="count"))
    return fig


def chart_mmbd_norm(results: dict) -> go.Figure:
    scores = results.get("normalized_scores", [])
    thresholds = results.get("thresholds", {}).get("normalized_score", {})
    n = len(scores)
    labels = [f"class {i}" for i in range(n)]

    def score_color(v):
        if v >= 5.0:   return COLORS["bar_bad"]
        if v >= 3.0:   return COLORS["bar_mild"]
        if v >= 1.5:   return "rgba(230,81,0,0.45)"
        return COLORS["bar_normal"]

    colors = [score_color(s) for s in scores]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=scores,
        marker_color=colors,
        hovertemplate="%{x}<br>score: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="#c62828", line_width=1,
                  annotation_text="suspicious", annotation_font_size=10, annotation_font_color="#c62828")
    fig.add_hline(y=1.5, line_dash="dot", line_color="#e65100", line_width=1,
                  annotation_text="mild", annotation_font_size=10, annotation_font_color="#e65100")
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def chart_mmbd_raw(results: dict) -> go.Figure:
    scores = results.get("per_class_scores", [])
    labels = [f"class {i}" for i in range(len(scores))]

    fig = go.Figure(go.Bar(
        x=labels, y=scores,
        marker_color=COLORS["neutral"],
        hovertemplate="%{x}<br>score: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200)
    return fig


def chart_aeva_anomaly(results: dict) -> go.Figure:
    """Per-class anomaly index bar chart."""
    anomaly_index = results.get("anomaly_index", [])
    if not anomaly_index:
        return None
    threshold = results.get("thresholds", {}).get("anomaly_index_threshold", 4.0)
    n = len(anomaly_index)
    labels = [f"class {i}" for i in range(n)]

    def bar_color(v):
        if v >= threshold:      return COLORS["bar_bad"]
        if v >= threshold * 0.5: return COLORS["bar_mild"]
        return COLORS["bar_normal"]

    colors = [bar_color(v) for v in anomaly_index]
    fig = go.Figure(go.Bar(
        x=labels, y=anomaly_index,
        marker_color=colors,
        hovertemplate="%{x}<br>anomaly index: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#c62828", line_width=1.5,
                  annotation_text=f"threshold {threshold}", annotation_font_size=10,
                  annotation_font_color="#c62828")
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def chart_aeva_heatmap(results: dict) -> go.Figure:
    """Source→target mean L2 distance heatmap. Low values = short adversarial path = suspicious."""
    matrix = results.get("source_target_mean_l2", [])
    if not matrix:
        return None
    n = len(matrix)
    labels = [f"class {i}" for i in range(n)]

    # Replace None with 0 for display
    z = [[v if v is not None else 0.0 for v in row] for row in matrix]

    # Annotate the suspected target column if present
    suspected = results.get("suspected_target")

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#c62828"],   # low L2 = short path = red = suspicious
            [0.4, "#ef9a9a"],
            [0.7, "#f5f5f5"],
            [1.0, "#1565c0"],   # high L2 = long path = blue = normal
        ],
        hovertemplate="source: %{y}<br>target: %{x}<br>L2 distance: %{z:.2f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="L2 dist", font=dict(size=11, family="IBM Plex Mono, monospace")),
            thickness=12,
            len=0.8,
        ),
    ))

    # Highlight suspected target column with a box
    if suspected is not None and 0 <= suspected < n:
        fig.add_shape(
            type="rect",
            x0=suspected - 0.5, x1=suspected + 0.5,
            y0=-0.5, y1=n - 0.5,
            line=dict(color="#c62828", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=suspected, y=-0.9,
            text="suspected target",
            showarrow=False,
            font=dict(size=10, color="#c62828", family="IBM Plex Mono, monospace"),
            xref="x", yref="y",
        )

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=380,
        xaxis=dict(showgrid=False, showline=False, zeroline=False, title="target class",
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, showline=False, zeroline=False, title="source class",
                   tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


def chart_aeva(results: dict) -> go.Figure:
    """Kept for backward compatibility — delegates to anomaly chart."""
    return chart_aeva_anomaly(results)


def chart_freeeagle(results: dict) -> go.Figure:
    tendency = results.get("tendency_per_target") or []
    if not tendency:
        return None
    labels = [f"class {i}" for i in range(len(tendency))]
    threshold = results.get("thresholds", {}).get("anomaly_metric_threshold", 2.0)
    colors = [COLORS["bar_bad"] if v > threshold else COLORS["bar_normal"] for v in tendency]

    fig = go.Figure(go.Bar(x=labels, y=tendency, marker_color=colors,
                           hovertemplate="%{x}<br>tendency: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#c62828", line_width=1.5,
                  annotation_text=f"threshold {threshold}", annotation_font_size=10,
                  annotation_font_color="#c62828")
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def chart_freeeagle_heatmap(results: dict) -> go.Figure:
    """Source→target anomaly matrix heatmap."""
    matrix = results.get("anomaly_matrix", [])
    if not matrix:
        return None
    n = len(matrix)
    labels = [f"class {i}" for i in range(n)]

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#f5f5f5"],
            [0.5, "#ef9a9a"],
            [1.0, "#c62828"],
        ],
        hovertemplate="source: %{y}<br>target: %{x}<br>score: %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title=dict(text="score", font=dict(size=11, family="IBM Plex Mono, monospace")),
            thickness=12, len=0.8,
        ),
    ))
    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=380,
        xaxis=dict(showgrid=False, showline=False, zeroline=False, title="target class",
                   tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, showline=False, zeroline=False, title="source class",
                   tickfont=dict(size=10), autorange="reversed"),
    )
    return fig


def chart_comparison(reports: dict) -> go.Figure:
    """Cross-defense comparison — one row per defense."""
    labels, verdicts, colors_list = [], [], []
    for d, rep in reports.items():
        if rep is None:
            continue
        v = rep.get("results", {}).get("verdict", "unknown")
        labels.append(d.upper())
        verdicts.append(v)
        colors_list.append(COLORS["bar_clean"] if is_clean(v) else COLORS["bar_bad"])

    if not labels:
        return None

    fig = go.Figure(go.Bar(
        x=labels,
        y=[1] * len(labels),
        marker_color=colors_list,
        text=verdicts,
        textposition="inside",
        textfont=dict(size=11, family="IBM Plex Mono, monospace"),
        hoverinfo="skip",
    ))
    base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **base,
        height=90,
        showlegend=False,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False),
        bargap=0.3,
    )
    return fig


# ── Defense result panels ──────────────────────────────────────────────────
def panel_strip(rep: dict):
    r = rep.get("results", {})
    stats = r.get("statistics", {})
    verdict = r.get("verdict", "unknown")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "average randomness",
        f"{stats.get('entropy_mean', 0):.3f}",
        help="Mean entropy across samples. Backdoored models produce unnaturally consistent (low entropy) outputs when perturbed — the trigger forces one prediction.",
    )
    c2.metric(
        "variability",
        f"{stats.get('entropy_std', 0):.3f}",
        help="Standard deviation of entropy. Low variability with low mean is a stronger backdoor signal.",
    )
    c3.metric("lowest score",  f"{stats.get('entropy_min', 0):.3f}")
    c4.metric("highest score", f"{stats.get('entropy_max', 0):.3f}")

    # Warn if threshold was auto-scaled (num_classes > 10 suggests non-CIFAR model)
    thr = r.get("thresholds", {}).get("entropy_mean_threshold", 0.45)
    params = r.get("parameters", {})
    dataset = r.get("dataset", "")
    if "imagenet" in dataset.lower() or "hf" in dataset.lower():
        st.info(
            f"Threshold auto-scaled to {thr:.3f} based on number of output classes. "
            "STRIP entropy scales with class count — results on large-vocabulary models should be interpreted carefully.",
            icon=None,
        )

    st.markdown("**Randomness score per sample** — low scores (red) indicate abnormally consistent model behavior, a backdoor signal")
    st.plotly_chart(chart_strip(r), use_container_width=True, config={"displayModeBar": False})

    st.markdown("**Entropy distribution**")
    st.plotly_chart(chart_strip_dist(r), use_container_width=True, config={"displayModeBar": False})


def panel_mmbd(rep: dict):
    r = rep.get("results", {})
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "statistical confidence",
        f"{r.get('p_value', 0):.4f}",
        help="p-value: how likely the score distribution would occur by chance. Below 0.05 suggests a backdoor is present.",
    )
    c2.metric(
        "activation dominance",
        f"{r.get('top_eigenvalue', 0):.2f}",
        help="A high value means one class overwhelmingly dominates the model's internal activations — a strong signal of backdoor manipulation.",
    )
    c3.metric(
        "classes analyzed",
        str(len(r.get("per_class_scores", []))),
        help="Number of output classes the defense inspected.",
    )

    st.markdown("**Suspicion score per class** — scores above 3.0 are suspicious, above 5.0 are very suspicious")
    st.plotly_chart(chart_mmbd_norm(r), use_container_width=True, config={"displayModeBar": False})

    st.markdown("**Raw activation scores per class**")
    st.plotly_chart(chart_mmbd_raw(r), use_container_width=True, config={"displayModeBar": False})


def panel_aeva(rep: dict):
    r = rep.get("results", {})

    # anomaly_index is a list in AEVA reports — take the max as the headline number
    anomaly_index_arr = r.get("anomaly_index", [])
    suspicion_score = r.get("suspicion_score") or (max(anomaly_index_arr) if anomaly_index_arr else 0.0)
    threshold = r.get("thresholds", {}).get("anomaly_index_threshold", 4.0)
    suspected_target = r.get("suspected_target")
    clean_acc = r.get("clean_accuracy")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "peak suspicion score",
        f"{suspicion_score:.3f}",
        help="The highest anomaly index across all classes. Above the threshold means a backdoor shortcut was detected.",
    )
    c2.metric(
        "detection threshold",
        f"{threshold:.1f}",
        help="Scores above this value are flagged as suspicious.",
    )
    if suspected_target is not None:
        c3.metric(
            "suspected target class",
            str(suspected_target),
            help="The class the backdoor is most likely steering predictions toward.",
        )
    else:
        c3.metric("suspected target class", "none")
    if clean_acc is not None:
        c4.metric(
            "clean accuracy",
            f"{clean_acc:.1%}",
            help="Model accuracy on unmodified test samples.",
        )

    st.markdown("**Anomaly index per class** — bars above threshold indicate a backdoor shortcut to that class")
    fig_bar = chart_aeva_anomaly(r)
    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "**Adversarial distance heatmap** — how far inputs must travel to cross from each source class (rows) "
        "to each target class (columns). Red = short path = easy to fool = suspicious"
    )
    fig_heat = chart_aeva_heatmap(r)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<p class="mono">distance matrix not in report</p>', unsafe_allow_html=True)


def panel_freeeagle(rep: dict):
    r = rep.get("results", {})
    params = r.get("parameters", {})

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "trigger strength",
        f"{r.get('anomaly_metric', 0):.4f}",
        help="How strongly a synthetic trigger steers the model toward one target class. High values suggest a backdoor trigger was implanted.",
    )
    c2.metric(
        "detection threshold",
        f"{r.get('thresholds', {}).get('anomaly_metric_threshold', 2.0):.2f}",
        help="Scores above this threshold are flagged as likely backdoored.",
    )
    c3.metric(
        "classes inspected",
        str(params.get("num_classes", "—")),
        help="Number of output classes analyzed by FreeEagle.",
    )

    st.markdown("**Trigger tendency per class** — high scores mean a synthetic trigger strongly steers predictions toward that class")
    fig = chart_freeeagle(r)
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        "**Source→target anomaly matrix** — how strongly each source class (rows) is redirected "
        "toward each target class (columns) by an optimized trigger. Bright cells indicate suspected backdoor paths."
    )
    fig_heat = chart_freeeagle_heatmap(r)
    if fig_heat:
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<p class="mono">anomaly matrix not in report</p>', unsafe_allow_html=True)


PANEL_FNS = {
    "strip":     panel_strip,
    "mmbd":      panel_mmbd,
    "aeva":      panel_aeva,
    "freeeagle": panel_freeeagle,
}


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Mithridatium")
    st.markdown('<p style="color:#555;font-size:12px;margin-bottom:24px;">backdoor detection</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model**")
    provider = st.selectbox("provider", PROVIDERS, label_visibility="collapsed")

    if provider == "torchvision":
        uploaded = st.file_uploader("upload checkpoint (.pth / .pt)", type=["pth", "pt"])
        model_path_input = st.text_input("or enter path", placeholder="models/resnet18.pth")
    else:
        uploaded = None
        hf_id = st.text_input("hugging face model id", value="microsoft/resnet-50")

    dataset = st.selectbox("dataset", DATASETS)

    st.markdown("**Defenses**")
    selected_defenses = []
    for d in DEFENSES:
        if st.checkbox(d, value=(d in ["mmbd", "strip"])):
            selected_defenses.append(d)

    st.markdown("**Load report JSON**")
    uploaded_reports = st.file_uploader(
        "drop existing reports to visualize",
        type=["json"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    run_btn = st.button("Run detection")

    if st.session_state.get("reports"):
        st.markdown("---")
        if st.button("Clear all reports", type="secondary"):
            st.session_state.reports = {}
            st.rerun()


# ── Session state ──────────────────────────────────────────────────────────
if "reports" not in st.session_state:
    st.session_state.reports = {}

if "logs" not in st.session_state:
    st.session_state.logs = []


# ── Load uploaded JSON reports ─────────────────────────────────────────────
if uploaded_reports:
    for f in uploaded_reports:
        try:
            rep = json.load(f)
            # Use filename (without extension) as the tab label key
            tab_key = Path(f.name).stem
            rep["_tab_label"] = tab_key
            st.session_state.reports[tab_key] = rep
        except Exception as e:
            st.error(f"Could not parse {f.name}: {e}")


# ── Run detection ──────────────────────────────────────────────────────────
if run_btn and selected_defenses:
    # Resolve model path
    if provider == "torchvision":
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            model_ref = tmp.name
            model_display_name = Path(uploaded.name).stem  # use original filename for display
        elif model_path_input:
            try:
                _validated = validate_checkpoint_path(model_path_input)
            except ValueError as _ve:
                st.error(f"Invalid model path: {_ve}")
                st.stop()
            model_ref = str(_validated)
            model_display_name = _validated.stem
        else:
            st.error("Provide a model file or path.")
            st.stop()
    else:
        model_ref = hf_id
        model_display_name = hf_id

    progress_placeholder = st.empty()
    logs: list[str] = []

    def log(msg):
        logs.append(msg)
        progress_placeholder.markdown(
            '<div class="status-log">' +
            "<br>".join(f"› {l}" for l in logs[-12:]) +
            "</div>",
            unsafe_allow_html=True,
        )

    log(f"model: {model_ref}")
    log(f"dataset: {dataset}")
    log(f"defenses: {', '.join(selected_defenses)}")
    log("─" * 40)

    new_reports = {}
    start = time.time()

    # Run defenses sequentially (parallel threading conflicts with Streamlit context)
    for d in selected_defenses:
        log(f"[{d}] starting...")
        try:
            rep = run_detection_cached(model_ref, dataset, d, display_name=model_display_name)
            rep["_tab_label"] = d
            new_reports[d] = rep
            verdict = rep.get("results", {}).get("verdict", "unknown")
            log(f"[{d}] done → {verdict}")
        except Exception as ex:
            log(f"[{d}] ERROR: {ex}")
            new_reports[d] = None

    elapsed = time.time() - start
    log(f"─" * 40)
    log(f"completed in {elapsed:.1f}s")

    st.session_state.reports.update(new_reports)
    st.session_state.logs = logs


# ── Main display ───────────────────────────────────────────────────────────
reports = st.session_state.reports

if not reports:
    st.markdown("""
    <div style="text-align:center; padding: 80px 0; color:#aaa;">
        <p style="font-family:'IBM Plex Mono',monospace; font-size:14px; letter-spacing:0.1em;">
            MITHRIDATIUM
        </p>
        <p style="font-size:13px; margin-top:8px; color:#bbb;">
            Upload a model and run detection, or drop existing report JSON files in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Group all reports by dataset
from collections import defaultdict
dataset_groups: dict[str, dict] = defaultdict(dict)
for k, rep in reports.items():
    if rep is None:
        continue
    ds = rep.get("dataset") or rep.get("results", {}).get("dataset", "unknown")
    dataset_groups[ds][k] = rep

# ── Dataset-level tabs ─────────────────────────────────────────────────────
dataset_keys = list(dataset_groups.keys())

if not dataset_keys:
    st.markdown(
        """
        <div style="text-align:center; padding: 80px 0; color:#aaa;">
            <p style="font-family:'IBM Plex Mono',monospace; font-size:14px; letter-spacing:0.1em;">
                No valid reports available.
            </p>
            <p style="font-size:13px; margin-top:8px; color:#bbb;">
                Remove errored reports or upload a new report JSON.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if len(dataset_keys) == 1:
    dataset_tabs = [st.container()]
else:
    dataset_tabs = st.tabs(dataset_keys)

for dataset_tab, dataset_key in zip(dataset_tabs, dataset_keys):
    group_reports = dataset_groups[dataset_key]
    with dataset_tab:

        # ── Banner — reflects all reports in this dataset group ───────────
        flagged = sum(
            1 for r in group_reports.values()
            if not is_clean(r.get("results", {}).get("verdict", ""))
        )
        total = len(group_reports)

        # List unique models in this group for the banner subtitle
        unique_models = list(dict.fromkeys(
            r.get("model_path", "unknown") for r in group_reports.values()
        ))
        def _short(mp: str) -> str:
            if "/" in mp and not mp.startswith(("C:\\", "/", ".")):
                return mp.split("/")[-1]
            return Path(mp).stem or mp
        models_display = ", ".join(_short(m) for m in unique_models)

        overall_verdict = "likely backdoored" if flagged > 0 else "likely clean"
        st.markdown(
            verdict_banner(overall_verdict, models_display, dataset_key, flagged, total),
            unsafe_allow_html=True,
        )

        # ── Defense overview (only when 2+ reports in this dataset group) ──
        if total > 1:
            fig_cmp = chart_comparison(group_reports)
            if fig_cmp:
                st.markdown(
                    f'<p class="section-header">defense overview — {dataset_key}</p>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})
                st.markdown("---")

        # ── Per-report tabs ───────────────────────────────────────────────
        available_keys = [k for k, v in group_reports.items() if v is not None]
        if available_keys:
            tab_labels = [group_reports[k].get("_tab_label", k) for k in available_keys]
            report_tabs = st.tabs(tab_labels)
            for report_tab, report_key in zip(report_tabs, available_keys):
                with report_tab:
                    rep = group_reports[report_key]
                    verdict = rep.get("results", {}).get("verdict", "unknown")
                    defense_key = rep.get("defense") or rep.get("results", {}).get("defense", "")
                    ts = rep.get("timestamp_utc", "")
                    model_p = rep.get("model_path", "")

                    hdr_col, badge_col, close_col = st.columns([3, 1, 1])
                    with hdr_col:
                        st.markdown(
                            f'<p class="mono">{_short(model_p)} &nbsp;·&nbsp; {ts}</p>',
                            unsafe_allow_html=True,
                        )
                    with badge_col:
                        st.markdown(
                            f'<div style="text-align:right">{verdict_badge(verdict)}</div>',
                            unsafe_allow_html=True,
                        )
                    with close_col:
                        if st.button("remove", key=f"close_{report_key}", type="secondary"):
                            del st.session_state.reports[report_key]
                            st.rerun()

                    panel_fn = PANEL_FNS.get(defense_key)
                    if panel_fn:
                        panel_fn(rep)
                    else:
                        st.json(rep.get("results", {}))

                    st.markdown("---")
                    render_trigger_impact_demo(rep)

                    with st.expander("raw report JSON"):
                        st.json(rep)

                    st.download_button(
                        label=f"download {report_key} report",
                        data=json.dumps(
                            {
                                k: v
                                for k, v in rep.items()
                                if k != "_tab_label" and not str(k).startswith("_ui_")
                            },
                            indent=2,
                        ),
                        file_name=f"mithridatium_{report_key}.json",
                        mime="application/json",
                    )
