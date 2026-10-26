import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from mithridatium.mcp.audit_agent import _load_mithridatium_mcp

REPAIR_TOOLS = {"run_clp", "run_lmr", "run_anp"}

REPAIR_SYSTEM_PROMPT = """
You repair backdoored image-classification checkpoints.
Tools: run_clp, run_lmr, run_anp.
Pass the model path the user gives. Never set out to the input model path.
Run only the methods they name. If they name none, run run_clp.
run_lmr needs lmr_target_class and lmr_prune_ratio. If the user did not give
both, do not call run_lmr and do not guess them; say which values are missing.
Report where the repaired checkpoint and its report were saved, and what the
tool result says was pruned. Do not invent metrics. A repaired checkpoint is
not proven clean.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

async def repair_agent(prompt: str):
    client = MultiServerMCPClient(_load_mithridatium_mcp())
    model = init_chat_model(model="gpt-4o-mini")
    tools = [tool for tool in await client.get_tools() if tool.name in REPAIR_TOOLS]
    agent = create_agent(model, tools, system_prompt=REPAIR_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": prompt})
    return result.get("messages", [])[-1].content

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Repair models/resnet18_poison.pth."
    print(asyncio.run(repair_agent(prompt)))
