import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent

from mithridatium.mcp.audit_agent import mithridatium_tools
from mithridatium.mcp.memory import with_memory
from mithridatium.mcp.progress import progress_config
from mithridatium.mcp.llm import chat_model, resolve_llm  # noqa: F401 (resolve_llm used by the CLI)

REPAIR_TOOLS = {"run_clp", "run_lmr", "run_anp"}

REPAIR_SYSTEM_PROMPT = """
You repair backdoored image-classification checkpoints.
Tools: run_clp, run_lmr, run_anp.
Pass the model path the user gives. Never set out to the input model path.
Run only the methods they name. If they name none, run run_clp.
run_lmr needs lmr_target_class. If the user did not give it, do not call
run_lmr and do not guess it; say which value is missing.
Report where the repaired checkpoint and its report were saved, and what the
tool result says was pruned. Do not invent metrics. A repaired checkpoint is
not proven clean.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_PROMPT = "Repair models/resnet18_poison.pth."

async def repair_agent(
    prompt: str, llm: str | None = None, memory=None, trace: list | None = None, progress: bool = False
):
    """Run the repair specialist. `memory` is a read-only Memory view; `trace`, if given,
    receives the agent's messages so the supervisor can record which tools ran."""
    model = chat_model(llm)
    async with mithridatium_tools(REPAIR_TOOLS) as tools:
        system_prompt, tools = with_memory(memory, "repair", prompt, REPAIR_SYSTEM_PROMPT, tools)
        agent = create_agent(model, tools, system_prompt=system_prompt)
        result = await agent.ainvoke({"messages": prompt}, config=progress_config(progress))
    messages = result.get("messages", [])
    if trace is not None:
        trace.extend(messages)
    return messages[-1].content

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    print(asyncio.run(repair_agent(prompt)))
