# MCP Server and Audit Agent

Mithridatium can expose the same audits as MCP tools, and a small agent can call those tools for one checkpoint. The tools do not reimplement the defenses. Each one calls `audit()` in `mithridatium/cli.py`.

The server lives in `mithridatium/mcp/`. The console script is `mithridatium-mcp`, which runs `mithridatium.mcp:main`.

## Install

The server and agent are an optional extra. Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[agent]"
```

The bundled agent calls `init_chat_model("gpt-4o-mini")`. That needs the OpenAI provider package and an API key in the same shell:

```bash
pip install langchain-openai
export OPENAI_API_KEY="sk-..."
```

Run the server and the agent on a machine where PyTorch imports successfully.

## Tools


| Tool            | Defense   |
| --------------- | --------- |
| `run_mmbd`      | MMBD      |
| `run_strip`     | STRIP     |
| `run_aeva`      | AEVA      |
| `run_freeeagle` | FreeEagle |



### Repair 
| Tool       | Method |
| ---------- | ------ |
| `run_clp`  | CLP    |
| `run_lmr`  | LMR    |
| `run_anp`  | ANP    |


Each tool takes the same general inputs as the CLI: `model`, `data`, `arch`, `provider`, and `hf_model_id`. Defense-specific options use the same names as the CLI flags, with underscores instead of hyphens (`freeeagle_anomaly_threshold`, `strip_mad_scale`, and so on).

Defaults match a local CIFAR-10 ResNet-18 checkpoint at `models/resnet18.pth`. Pass the checkpoint path you actually want to audit.

## Reports

A tool writes the report before it returns `{"report_path": ..., "report": ...}` to the caller:

```text
reports/<checkpoint-stem>_<defense>.json
```

`models/resnet18_poison.pth` audited with MMBD becomes `reports/resnet18_poison_mmbd.json`. The path is relative to the server process working directory. The agent sets that directory to the repository root. An existing file with the same name is overwritten.

The audit's own progress logs are not printed on the MCP connection. The saved JSON file and the tool result are the record.

## Run the Server

For an MCP client on the same machine:

```bash
mithridatium-mcp
```

On a cluster, shell startup functions are sometimes exported as `BASH_FUNC_*` environment variables. The MCP client warns on those values and can refuse to start the server. The bundled agent drops names that start with `BASH_FUNC_` before it launches the server. A hand-written client config should do the same.

## Run the Agent

The agent connects to the local server, asks the model to call the requested defenses, and prints the model's last message:

```bash
python -m mithridatium.mcp.audit_agent
```

That command audits `models/resnet18_poison.pth`. With no defense named, the system prompt tells the model to run MMBD only. To ask for something else, pass a prompt:

```bash
python -m mithridatium.mcp.audit_agent "Run STRIP and MMBD on models/resnet18_clean.pth"
```

The detect and repair agents work the same way:

```bash
python -m mithridatium.mcp.detect_agent "Run TED on models/resnet18_poison.pth"
python -m mithridatium.mcp.repair_agent "Run LMR on models/resnet18_poison.pth with target class 0 and prune ratio 0.05"
python -m mithridatium.mcp.repair_agent "Run ANP on models/resnet18_poison.pth"
```

With no prompt, the detect agent runs SCALE-UP and the repair agent runs CLP on `models/resnet18_poison.pth`. Each agent only receives its own tools. The repair agent supports `run_clp`, `run_lmr`, and `run_anp`. The detect agent's tools are `run_scaleup`, `run_ted` and `run_ibdpsc`, and their options use the CLI flag names with underscores (`ibdpsc_scale`, `ibdpsc_num_models`, and so on). From Python, call `audit_agent()`, `detect_agent()` or `repair_agent()` with the prompt string.

The printed reply is a short verdict summary. The full report is the JSON file under `reports/`.

### Supervisor Agent

The supervisor agent coordinates all three specialists (`audit`, `detect`, `repair`) from one prompt:

```bash
mithridatium agent "Check models/resnet18_poison.pth and repair it if it's poisoned"
```

Pass `--report` to save a JSON summary of the run:

```bash
mithridatium agent "Check models/resnet18_poison.pth and repair it if it's poisoned" --report reports/agent_run.json
```

Without `--report`, nothing extra is written to disk; the agent prints its final reply to stdout. Like `audit` and `repair`, an existing file is not overwritten without `--force`.

The summary is built from the tool calls and tool results in the agent's message history, not parsed from the model's final reply:

```json
{
  "prompt": "Check models/resnet18_poison.pth and repair it if it's poisoned",
  "agent": "supervisor",
  "chat_model": "gpt-4o-mini",
  "steps": [
    {
      "step": 1,
      "specialist": "audit",
      "request": "Audit models/resnet18_poison.pth",
      "report_paths": ["reports/resnet18_poison_mmbd.json"],
      "report_path": "reports/resnet18_poison_mmbd.json",
      "checkpoint_paths": [],
      "checkpoint_path": null,
      "verdicts": ["attack"],
      "verdict": "attack",
      "result": "..."
    }
  ],
  "final_message": "..."
}
```

## Contributor Notes

- Register new tools on the shared `mcp` object in `mithridatium/mcp/__init__.py`, and import the new module from `main()` before `mcp.run`. Importing a tool module at the top of `__init__.py` cycles, because the tool module imports `mcp` from the package.
- Keep `@mcp.tool(run_in_thread=False)`. The audit builds a PyTorch `DataLoader` with worker processes. Starting those workers from a FastMCP background thread deadlocks. Forking them from the server process can also deadlock, so `main()` sets `MITHRIDATIUM_NUM_WORKERS=0` and the loaders in `utils.py` run in-process.
- Stay on FastMCP 3.x. `langchain-mcp-adapters` 0.3.2 requires `mcp` 1.x, and FastMCP 4 requires `mcp` 2.x.



## Related Docs

- [CLI flow](cli-flow.md)
- [Reporting pipeline](reporting-pipeline.md)
- [Testing overview](../testing/overview.md)
- [Known issues](../handoff/known-issues.md)

