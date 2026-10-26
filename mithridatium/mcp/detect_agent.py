import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from mithridatium.mcp.audit_agent import _load_mithridatium_mcp

DETECT_TOOLS = {"run_scaleup", "run_ted", "run_ibdpsc"}

DETECT_SYSTEM_PROMPT = """
You run poisoning detection on image-classification checkpoints.
Tools: run_scaleup, run_ted, run_ibdpsc (IBD-PSC).
Pass the model path the user gives. Unless they say otherwise, use
data='cifar10'.
Run only the methods they name. If they name none, run run_scaleup.
Report each method's verdict, how many inputs it flagged, and its
report_path from the tool result. Do not invent metrics.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

async def detect_agent(prompt: str):
    client = MultiServerMCPClient(_load_mithridatium_mcp())
    model = init_chat_model(model="gpt-4o-mini")
    tools = [tool for tool in await client.get_tools() if tool.name in DETECT_TOOLS]
    agent = create_agent(model, tools, system_prompt=DETECT_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": prompt})
    return result.get("messages", [])[-1].content

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Run detection on models/resnet18_poison.pth."
    print(asyncio.run(detect_agent(prompt)))
