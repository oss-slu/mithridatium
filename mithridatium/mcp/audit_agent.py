import asyncio
import sys
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from mithridatium.mcp.llm import chat_model, resolve_llm  # noqa: F401 (resolve_llm used by the CLI)

AUDIT_TOOLS = {"run_mmbd", "run_strip", "run_aeva", "run_freeeagle"}

AUDIT_SYSTEM_PROMPT = """
You audit image-classification checkpoints for backdoors.
Tools: run_mmbd, run_strip, run_aeva, run_freeeagle.
Pass the model path the user gives. Unless they say otherwise, use
data='cifar10', arch='resnet18', provider='torchvision'.
Run only the defenses they name. If they name none, run run_mmbd.
Report each defense's verdict, its suspected_target if the report has one,
and its report_path from the tool result. Do not invent metrics.
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

DEFAULT_PROMPT = "Audit models/resnet18_poison.pth."

async def audit_agent(prompt: str, llm: str | None = None):
    client = MultiServerMCPClient(_load_mithridatium_mcp())
    model = chat_model(llm)
    tools = [tool for tool in await client.get_tools() if tool.name in AUDIT_TOOLS]
    agent = create_agent(model, tools, system_prompt=AUDIT_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": prompt})
    return result.get("messages", [])[-1].content

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    print(asyncio.run(audit_agent(prompt)))