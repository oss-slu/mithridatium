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

## MCP Package Versions

`langchain-mcp-adapters` 0.3.2 requires the `mcp` 1.x SDK. FastMCP 4 requires `mcp` 2.x, which removed `mcp.shared.context.RequestContext` and breaks the adapter import. Keep the `agent` extra on FastMCP 3.x (`fastmcp>=3.4,<4`).

## MCP Tool Threads

Audit tools are registered with `@mcp.tool(run_in_thread=False)`. The CLI dataloader uses `num_workers=2`. Starting those worker processes from a FastMCP background thread deadlocks, which showed up as a STRIP run that never wrote its report. Leave the tools on the server's main thread.

The bundled agent also drops environment variables whose names start with `BASH_FUNC_`. Cluster shells export Lmod and Spack functions under those names, and the MCP client warns or refuses to spawn the server when those values contain unexpanded `${...}` references.

## Legacy Service Path

`mithridatium/service.py` contains a service-style detection path that is not fully aligned with the newer CLI options, especially FreeEagle and Hugging Face details. Treat `mithridatium/cli.py` as the current source of truth for command-line behavior.
