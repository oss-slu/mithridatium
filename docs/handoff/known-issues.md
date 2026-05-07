# Known Issues

## Defense Calibration

Thresholds are still research defaults or implementation-specific heuristics. They should be calibrated with known clean and known backdoored benchmark models before being treated as operational decisions.

## Dataset Mismatch

Dataset and preprocessing mismatch can cause misleading results. This is especially important for STRIP, which relies on representative inputs and entropy under input mixing.

## FreeEagle Scope

FreeEagle is white-box, data-free, and currently supports known ResNet-family models in Mithridatium. It does not currently support arbitrary architectures or the Hugging Face wrapper.

## Hugging Face Scope

Hugging Face support is limited to models compatible with `AutoModelForImageClassification`. Architecture compatibility and preprocessing can vary by model repo.

## AEVA Runtime

AEVA is computationally expensive. Runtime grows with the number of source-target class pairs and HSJA query settings. Smoke tests should use reduced class ranges and small query budgets.

## MMBD Class Coverage

MMBD probes only a subset of classes by default. If a backdoor target is outside the probed set, the defense may miss it.

## Report Summaries

`render_summary()` has detailed summary branches for MMBD, STRIP, and FreeEagle. AEVA currently relies more on the generic fallback summary.

## Legacy Service Path

`mithridatium/service.py` contains a service-style detection path that is not fully aligned with the newer CLI options, especially FreeEagle and Hugging Face details. Treat `mithridatium/cli.py` as the current source of truth for command-line behavior.
