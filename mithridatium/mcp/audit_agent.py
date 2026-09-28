import asyncio
import sys
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

AUDIT_SYSTEM_PROMPT = """
You audit image-classification checkpoints for backdoors.
Tools: run_mmbd, run_strip, run_aeva, run_freeeagle.
Pass the model path the user gives. Unless they say otherwise, use
data='cifar10', arch='resnet18', provider='torchvision'.
Run only the defenses they name. If they name none, run run_mmbd.
Report each defense's verdict from the tool result. Do not invent metrics.
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

async def audit_agent(model_path: str):
    client = MultiServerMCPClient(_load_mithridatium_mcp())
    model = init_chat_model(model="gpt-4o-mini")
    tools = await client.get_tools()
    agent = create_agent(model, tools, system_prompt=AUDIT_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": f"Audit {model_path}."})
    return result.get("messages", [])[-1].content

if __name__ == "__main__":
    print(asyncio.run(audit_agent("models/resnet18_poison.pth")))