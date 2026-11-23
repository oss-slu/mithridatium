import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from mithridatium.mcp.audit_agent import audit_agent
from mithridatium.mcp.detect_agent import detect_agent
from mithridatium.mcp.llm import chat_model, resolve_llm
from mithridatium.mcp.memory import Memory, with_memory
from mithridatium.mcp.progress import progress_config
from mithridatium.mcp.repair_agent import repair_agent

def _specialist_tools(llm: str, memory: Memory | None = None):
    """Wrap the three specialist agents as tools that all use the same chat model.

    With memory, each specialist gets a read-only view and the supervisor
    records an episode of which tools the specialist ran and how they went.
    """

    async def call(name: str, run, request: str) -> str:
        if memory is None:
            return await run(request, llm=llm)
        trace: list[Any] = []
        try:
            reply = await run(request, llm=llm, memory=memory.read_only(), trace=trace)
        except Exception as ex:
            memory.record_episode(name, _episode_text(name, request, trace, error=ex))
            raise
        memory.record_episode(name, _episode_text(name, request, trace))
        return reply

    @tool
    async def audit(request: str) -> str:
        """Audit a checkpoint for backdoors (MMBD, STRIP, AEVA, FreeEagle). Include the model path."""
        return await call("audit", audit_agent, request)

    @tool
    async def detect(request: str) -> str:
        """Run poisoned-input detection (ScaleUp, TED, TED++, IBD-PSC) on a checkpoint. Include the model path."""
        return await call("detect", detect_agent, request)

    @tool
    async def repair(request: str) -> str:
        """Repair a backdoored checkpoint (CLP, LMR, ANP). Include the model path. Returns the repaired checkpoint path."""
        return await call("repair", repair_agent, request)

    return [audit, detect, repair]

SUPERVISOR_SYSTEM_PROMPT = """
You coordinate backdoor checks on image-classification checkpoints.
Tools: audit (backdoor defenses), detect (poisoned-input detection),
repair (prunes a checkpoint to remove a backdoor).
Each tool is a specialist agent. Send it a plain-English request with the
model path and any method names or settings the user gave.
Do what the user asks. If they just say "check" or "test" a model, run
audit, then detect.
Never run repair unless the user asked for it. If they asked for a repair
outright, run repair directly. If they made it conditional ("if it's
poisoned"), run repair only if at least one audit or detection result
flagged the model. If nothing flagged it, say so and stop.
After a repair, run audit on the repaired checkpoint path that repair
returned, not on the original.
If an audit result gives a suspected_target, include it as the target class
when you ask for an LMR repair. Pass through what a specialist says it is
missing (for example LMR's target class). Do not make those values up.
End with a short summary: each tool you ran, its verdict, and where the
repaired checkpoint and reports were saved. Report only what the tools
returned. Do not invent metrics. A repaired checkpoint is not proven clean.
Memory from past runs is context only. Current tool results always win.
Never report a past verdict as a current one.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# converts message content (blocks or str) into text
def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content) if content is not None else ""


# extracts report paths, checkpoint paths, and verdicts from tool outputs
def _extract_paths_and_verdict(content: Any) -> tuple[list[str], list[str], list[str]]:
    report_paths: list[str] = []
    checkpoint_paths: list[str] = []
    verdicts: list[str] = []

    # checks structured dict or json tool outputs first
    parsed_dict: dict[str, Any] | None = None
    if isinstance(content, dict):
        parsed_dict = content
    elif isinstance(content, str):
        content_stripped = content.strip()
        if (content_stripped.startswith("{") and content_stripped.endswith("}")) or (
            content_stripped.startswith("[") and content_stripped.endswith("]")
        ):
            try:
                parsed_json = json.loads(content_stripped)
                if isinstance(parsed_json, dict):
                    parsed_dict = parsed_json
            except Exception:
                pass

    if parsed_dict is not None:
        if "report_path" in parsed_dict and parsed_dict["report_path"]:
            report_paths.append(str(parsed_dict["report_path"]))
        if "checkpoint" in parsed_dict and parsed_dict["checkpoint"]:
            checkpoint_paths.append(str(parsed_dict["checkpoint"]))
        if "checkpoint_path" in parsed_dict and parsed_dict["checkpoint_path"]:
            checkpoint_paths.append(str(parsed_dict["checkpoint_path"]))
        if "verdict" in parsed_dict and parsed_dict["verdict"]:
            verdicts.append(str(parsed_dict["verdict"]))
        if "report" in parsed_dict and isinstance(parsed_dict["report"], dict):
            res = parsed_dict["report"].get("results", {})
            if isinstance(res, dict) and "verdict" in res and res["verdict"]:
                verdicts.append(str(res["verdict"]))

    text = str(content)

    # finds report paths in text excluding backticks
    for match in re.findall(r"[^\s,;'\"<>()`]+\.json", text):
        clean_match = match.strip("'\"`")
        if clean_match and clean_match not in report_paths:
            report_paths.append(clean_match)

    # finds checkpoint paths in text excluding backticks
    for match in re.findall(r"[^\s,;'\"<>()`]+\.(?:pth|pt)", text):
        clean_match = match.strip("'\"`")
        if clean_match and clean_match not in checkpoint_paths:
            checkpoint_paths.append(clean_match)

    # check report files on disk if available for objective verdict
    for r_path in report_paths:
        try:
            p = Path(r_path)
            if p.exists() and p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                v = data.get("results", {}).get("verdict")
                if v and str(v) not in verdicts:
                    verdicts.append(str(v))
        except Exception:
            pass

    # finds explicit verdict pattern from text if not in report file
    if not verdicts:
        verdict_match = re.search(r"verdict[:\s=]+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        if verdict_match:
            verdicts.append(verdict_match.group(1))

    return report_paths, checkpoint_paths, verdicts


# builds structured json summary from supervisor message history
def build_agent_run_summary(
    prompt: str,
    messages: list[Any],
    agent_name: str = "supervisor",
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    # index tool messages by tool_call_id
    tool_messages_by_id = {
        getattr(msg, "tool_call_id", None): msg
        for msg in messages
        if getattr(msg, "tool_call_id", None)
    }
    final_message = ""

    # extract final reply text from assistant messages
    for msg in messages:
        msg_type = getattr(msg, "type", None) or type(msg).__name__
        if msg_type in ("ai", "AIMessage"):
            content = getattr(msg, "content", "")
            text = _content_to_text(content)
            if text:
                final_message = text

    steps: list[dict[str, Any]] = []

    # extract ordered steps matching each tool call to its tool message
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue

        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            tc_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})

            if isinstance(tc_args, dict):
                request_sent = tc_args.get("request", "") or tc_args.get("prompt", "")
                if not request_sent:
                    request_sent = json.dumps(tc_args) if tc_args else ""
            else:
                request_sent = str(tc_args)

            # pair with tool result by tool_call_id
            tool_msg = tool_messages_by_id.get(tc_id)
            raw_result = getattr(tool_msg, "content", "") if tool_msg is not None else ""
            report_paths, checkpoint_paths, verdicts = _extract_paths_and_verdict(raw_result)

            # only list fields are stored for clean schema
            step_data: dict[str, Any] = {
                "step": len(steps) + 1,
                "specialist": tc_name,
                "request": request_sent,
                "report_paths": report_paths,
                "checkpoint_paths": checkpoint_paths,
                "verdicts": verdicts,
                "status": getattr(tool_msg, "status", "success") if tool_msg is not None else "no result",
                "result": raw_result if isinstance(raw_result, str) else json.dumps(raw_result),
            }
            steps.append(step_data)

    return {
        "prompt": prompt,
        "agent": agent_name,
        "chat_model": model_name,
        "steps": steps,
        "final_message": final_message,
    }


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _error_text(result: str) -> str:
    """The error lines of a failed tool result, without log noise or content-block JSON."""
    try:
        parsed = json.loads(result)
        if isinstance(parsed, list):
            result = _content_to_text(parsed)
    except (TypeError, ValueError):
        pass
    lines = [line.strip() for line in str(result).splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower() or line.lower().startswith("reason")]
    return " ".join(errors[-2:] or lines[-1:])


def _episode_text(agent: str, request: str, messages: list[Any], error: Exception | None = None) -> str:
    """One-line record of a run: each tool called, whether it worked, its verdicts and paths.

    Built only from tool results. The model's own reply is left out unless no
    tool ran (then it is the only record, e.g. a missing LMR target class),
    because a stored reply gets recalled and repeated in later runs.
    """
    summary = build_agent_run_summary(request, messages, agent_name=agent)
    steps = [step for step in summary["steps"] if not _is_memory_tool(step["specialist"])]
    parts = []
    for step in steps:
        detail = []
        if step["status"] != "success":
            detail.append(f"{step['status']}: {_clip(_error_text(step['result']), 160)}")
        if step["verdicts"]:
            detail.append("verdict " + ", ".join(step["verdicts"]))
        if step["checkpoint_paths"]:
            detail.append("checkpoint " + ", ".join(step["checkpoint_paths"]))
        if step["report_paths"]:
            detail.append("report " + ", ".join(step["report_paths"]))
        ok = "ok" if step["status"] == "success" else "FAILED"
        parts.append(f"{step['specialist']} {ok}" + (f" ({'; '.join(detail)})" if detail else ""))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"[{stamp}] {agent} asked {_clip(request)!r} -> " + ("; ".join(parts) or "no tools called")
    if error is not None:
        text += f". Run failed: {_clip(error, 160)}"
    elif not steps and summary["final_message"]:
        text += f". Replied: {_clip(summary['final_message'])}"
    return text


def _is_memory_tool(name: str) -> bool:
    return name.startswith(("search_", "manage_")) and name.endswith("_memory")


def _called_tools(messages: list[Any]) -> bool:
    """True if any non-memory tool was called."""
    return any(
        not _is_memory_tool(tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""))
        for m in messages
        for tc in (getattr(m, "tool_calls", None) or [])
    )


DEFAULT_PROMPT = "Run a test suite (audit and detect) as needed on models/resnet18_poison.pth and run repairs if there is sufficient evidence of a backdoor."

async def supervisor_agent(
    prompt: str,
    llm: str | None = None,
    return_summary: bool = False,
    memory: bool = True,
    progress: bool = False,
):
    # resolve once so the supervisor and every specialist it calls use one model
    llm = resolve_llm(llm)
    model = chat_model(llm)
    store = Memory.open() if memory else None
    try:
        system_prompt, tools = with_memory(
            store, "supervisor", prompt, SUPERVISOR_SYSTEM_PROMPT, _specialist_tools(llm, store)
        )
        agent = create_agent(model, tools, system_prompt=system_prompt)
        result = await agent.ainvoke({"messages": prompt}, config=progress_config(progress))
        messages = result.get("messages", [])
        # a run with no tool calls (a question about memory, a chat reply) is not an
        # episode; recording it would feed the model's answer back as history
        if store is not None and _called_tools(messages):
            store.record_episode("supervisor", _episode_text("supervisor", prompt, messages))
    finally:
        if store is not None:
            store.save()
    if return_summary:
        summary = build_agent_run_summary(
            prompt=prompt,
            messages=messages,
            agent_name="supervisor",
            model_name=llm,
        )
        return summary["final_message"], summary
    return _content_to_text(messages[-1].content if messages and hasattr(messages[-1], "content") else "")


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or DEFAULT_PROMPT
    print(asyncio.run(supervisor_agent(prompt)))