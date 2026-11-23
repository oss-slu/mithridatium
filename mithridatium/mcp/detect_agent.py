import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent

from mithridatium.mcp.audit_agent import mithridatium_tools
from mithridatium.mcp.memory import with_memory
from mithridatium.mcp.progress import progress_config
from mithridatium.mcp.llm import chat_model, resolve_llm  # noqa: F401 (resolve_llm used by the CLI)

DETECT_TOOLS = {"run_scaleup", "run_ted", "run_ibdpsc", "run_tedpp"}

DETECT_SYSTEM_PROMPT = """
You run poisoning detection on image-classification checkpoints.
Tools: run_scaleup, run_ted, run_ibdpsc (IBD-PSC), run_tedpp.
Pass the model path the user gives. Unless they say otherwise, use
data='cifar10'.
Run only the methods they name. If they name none, run run_scaleup.
Report each method's verdict, how many inputs it flagged, and its
report_path from the tool result. Do not invent metrics.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_PROMPT = "Run detection on models/resnet18_poison.pth."

async def detect_agent(
    prompt: str, llm: str | None = None, memory=None, trace: list | None = None, progress: bool = False
):
    """Run the detect specialist. `memory` is a read-only Memory view; `trace`, if given,
    receives the agent's messages so the supervisor can record which tools ran."""
    model = chat_model(llm)
    async with mithridatium_tools(DETECT_TOOLS) as tools:
        system_prompt, tools = with_memory(memory, "detect", prompt, DETECT_SYSTEM_PROMPT, tools)
        agent = create_agent(model, tools, system_prompt=system_prompt)
        result = await agent.ainvoke({"messages": prompt}, config=progress_config(progress))
    messages = result.get("messages", [])
    if trace is not None:
        trace.extend(messages)
    return messages[-1].content

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    print(asyncio.run(detect_agent(prompt)))
