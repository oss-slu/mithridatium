import asyncio
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from mithridatium.mcp.audit_agent import audit_agent
from mithridatium.mcp.detect_agent import detect_agent
from mithridatium.mcp.repair_agent import repair_agent

@tool
async def run_audit(request: str) -> str:
    """Audit a checkpoint for backdoors (MMBD, STRIP, AEVA, FreeEagle). Include the model path."""
    return await audit_agent(request)

@tool
async def run_detect(request: str) -> str:
    """Run poisoned-input detection (ScaleUp, TED, IBD-PSC) on a checkpoint. Include the model path."""
    return await detect_agent(request)

@tool
async def run_repair(request: str) -> str:
    """Repair a backdoored checkpoint (CLP, LMR, ANP). Include the model path. Returns the repaired checkpoint path."""
    return await repair_agent(request)

SUPERVISOR_SYSTEM_PROMPT = """
You coordinate backdoor checks on image-classification checkpoints.
Tools: run_audit (backdoor defenses), run_detect (poisoned-input detection),
run_repair (prunes a checkpoint to remove a backdoor).
Each tool is a specialist agent. Send it a plain-English request with the
model path and any method names or settings the user gave.
Do what the user asks. If they just say "check" or "test" a model, run
run_audit, then run_detect.
Run run_repair only if the user asked for repairs and at least one audit or
detection result flagged the model. If nothing flagged it, say so and stop.
After a repair, run run_audit on the repaired checkpoint path that run_repair
returned, not on the original.
Pass through what a specialist says it is missing (for example LMR's target
class and prune ratio). Do not make those values up.
End with a short summary: each tool you ran, its verdict, and where the
repaired checkpoint and reports were saved. Report only what the tools
returned. Do not invent metrics. A repaired checkpoint is not proven clean.
"""


def _server_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("BASH_FUNC_")
    }

def _load_mithridatium_mcp():
    return {
        "mithridatium": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-c", "from mithridatium.mcp import main; main()"],
            "cwd": Path(__file__).resolve().parents[2],
            "env": _server_env(),
        }
    }

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

async def supervisor_agent(prompt: str):
    model = init_chat_model(model="gpt-4o-mini")
    agent = create_agent(model, [run_audit, run_detect, run_repair], system_prompt=SUPERVISOR_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": prompt})
    return result.get("messages", [])[-1].content


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Run a test suite (audit and detect) as needed on models/resnet18_poison.pth and run repairs as needed."
    print(asyncio.run(supervisor_agent(prompt)))