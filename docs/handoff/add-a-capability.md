# Add a Capability

A checklist of every file a new capability touches. Use it when you open a PR that adds a model-level defense (`audit --defense`), an input-level method (`detect --method`), a repair method (`repair --method`), or a whole new CLI verb.

Most of these lists are kept by hand, so a missed file usually doesn't fail loudly. It just leaves the CLI, the UI, and the docs out of sync.

## A. New `audit --defense`

| What | Where |
| --- | --- |
| Defense module | `mithridatium/defenses/<name>.py`, exposing a `run_<name>(model, config, device=...)` that returns a `dict` of results (see `run_mmbd` in `mmbd.py`) |
| CLI allow-list | `DEFENSES` in `mithridatium/cli.py` |
| CLI dispatch | the `if d == ...` chain in `audit()` in `mithridatium/cli.py`, plus the import at the top of the file |
| CLI flags | prefix every defense-specific option with `--<name>-` (for example `--strip-mad-scale`) |
| Model compatibility | `ensure_defense_compatibility` in `mithridatium/loader.py`. Add your defense to the logits-only set, or add a branch that raises `ValueError` when the model lacks what you need (feature module, gradients, and so on). If you skip this step, every run fails with `Unsupported defense`. |
| Report schema | `reports/report_schema.json`. Add an `if`/`then` block under `allOf` keyed on `"defense": {"const": "<name>"}` that describes your `results` shape. |
| Terminal summary | an `if defense == "<name>"` block in `render_summary` in `mithridatium/report.py` |
| Other registries | `DEFENSES` in `mithridatium/service.py` (feeds the Gradio UI) and `DEFENSES` in `app.py` (Streamlit) |
| MCP tool | a `run_<name>` tool in `mithridatium/mcp/audit_mcp.py` |
| Tests | a unit test in `tests/unit/test_<name>.py`; a report-schema contract test in `tests/integration/test_report_schema.py`; add `"<name>": "--<name>-"` to `DEFENSE_PREFIXES` in `tests/unit/test_cli_docs_sync.py`; update `test_cli_defenses_lists_current_supported_defenses` in `tests/smoke/test_cli_smoke.py` |
| Docs | `docs/defenses/<name>.md` listing every `--<name>-*` flag (the docs-sync test fails otherwise); a row in `docs/defenses/overview.md`; links in `docs/README.md` and the "Supported Defenses" section of `docs/architecture/cli-flow.md`; an example command in `docs/testing/sample-commands.md` |

## B. New `detect --method` (input-level)

Same shape as section A, with these swaps:

- Module goes in `mithridatium/detect/<name>.py`.
- Allow-list is `DETECT_METHODS` in `mithridatium/cli.py`, and dispatch lives in `detect()`.
- Schema block is keyed on `"defense": {"const": "<name>"}` with `"mode": "input-level"` in `results` (see the `scaleup` block).
- MCP tool goes in `mithridatium/mcp/detect_mcp.py`.
- Docs go in `docs/defenses/<name>.md` and `docs/detect.md`.
- `ensure_defense_compatibility` is not called by `detect`.

## C. New `repair --method`

- Module goes in `mithridatium/repair/<name>.py`. Re-export it from `mithridatium/repair/__init__.py` if other code imports it from there.
- Allow-list is `REPAIR_METHODS` in `mithridatium/cli.py`. Dispatch lives in `repair()`, which follows the same validate → load → run → report → write steps as LMR.
- Build the report with `build_repair_report` (it sets `method`, not `defense`), then call `validate_report_data`.
- Add a schema `if`/`then` block for the repair results.
- MCP tool goes in `mithridatium/mcp/repair_mcp.py`.
- Tests: a unit test (see `tests/unit/test_clp.py`), plus smoke tests for missing or invalid flags (see the `test_cli_repair_*` tests in `tests/smoke/test_cli_smoke.py`).
- Docs: a section in `docs/repair.md` with requirements, an example command, and caveats.

## D. New CLI verb

Use this when the capability doesn't fit `audit`, `detect`, or `repair`.

- Add an `@app.command()` function in `mithridatium/cli.py`. Reuse `_check_out_writable`, `_write_json`, and the `EXIT_*` codes so the exit codes match the other commands.
- If the verb has more than one method, give it its own `<VERB>_METHODS` set and validate against it before doing any work, returning `EXIT_USAGE_ERROR`.
- Pick the report shape. Use `build_report` (keyed on `defense`) or `build_repair_report` (keyed on `method`); the schema's top-level `anyOf` requires exactly one of those two keys. Add the matching `allOf` block.
- Add a `--help` smoke test and an unknown-method test in `tests/smoke/test_cli_smoke.py`.
- Add `docs/<verb>.md`, link it from `docs/README.md`, and add the verb to "Basic CLI Commands" in `docs/architecture/cli-flow.md`.
- If agents should be able to call it, add an MCP server next to the existing ones in `mithridatium/mcp/`.

## Before opening the PR

```bash
pytest tests/unit tests/smoke
codespell
mithridatium <verb> --help
```

Paste one real CLI command and its output (or the JSON report it wrote) into the PR description.
