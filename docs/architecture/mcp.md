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


Each tool takes the same general inputs as the CLI: `model`, `data`, `arch`, `provider`, and `hf_model_id`. Defense-specific options use the same names as the CLI flags, with underscores instead of hyphens (`freeeagle_anomaly_threshold`, `strip_mad_scale`, and so on).

Defaults match a local CIFAR-10 ResNet-18 checkpoint at `models/resnet18.pth`. Pass the checkpoint path you actually want to audit.

## Reports

A tool writes the report before it returns the JSON to the caller:

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

That command audits `models/resnet18_poison.pth`. With no defense named, the system prompt tells the model to run MMBD only. To audit another checkpoint, call `audit_agent()` from `mithridatium.mcp.audit_agent` with that path.

The printed reply is a short verdict summary. The full report is the JSON file under `reports/`.

## Contributor Notes

- Register new tools on the shared `mcp` object in `mithridatium/mcp/__init__.py`, and import the new module from `main()` before `mcp.run`. Importing a tool module at the top of `__init__.py` cycles, because the tool module imports `mcp` from the package.
- Keep `@mcp.tool(run_in_thread=False)`. The audit builds a PyTorch `DataLoader` with worker processes. Starting those workers from a FastMCP background thread deadlocks.
- Stay on FastMCP 3.x. `langchain-mcp-adapters` 0.3.2 requires `mcp` 1.x, and FastMCP 4 requires `mcp` 2.x.



## Related Docs

- [CLI flow](cli-flow.md)
- [Reporting pipeline](reporting-pipeline.md)
- [Testing overview](../testing/overview.md)
- [Known issues](../handoff/known-issues.md)

