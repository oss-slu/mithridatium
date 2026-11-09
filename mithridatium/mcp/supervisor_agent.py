import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from mithridatium.mcp.audit_agent import audit_agent
from mithridatium.mcp.detect_agent import detect_agent
from mithridatium.mcp.repair_agent import repair_agent

@tool
async def audit(request: str) -> str:
    """Audit a checkpoint for backdoors (MMBD, STRIP, AEVA, FreeEagle). Include the model path."""
    return await audit_agent(request)

@tool
async def detect(request: str) -> str:
    """Run poisoned-input detection (ScaleUp, TED, IBD-PSC) on a checkpoint. Include the model path."""
    return await detect_agent(request)

@tool
async def repair(request: str) -> str:
    """Repair a backdoored checkpoint (CLP, LMR, ANP). Include the model path. Returns the repaired checkpoint path."""
    return await repair_agent(request)

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
missing (for example LMR's prune ratio). Do not make those values up.
End with a short summary: each tool you ran, its verdict, and where the
repaired checkpoint and reports were saved. Report only what the tools
returned. Do not invent metrics. A repaired checkpoint is not proven clean.
"""

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _extract_paths_and_verdict(content: Any) -> tuple[list[str], list[str], list[str]]:
    # pull report paths checkpoint paths and verdicts from tool message
    report_paths: list[str] = []
    checkpoint_paths: list[str] = []
    verdicts: list[str] = []

    # parses structured dict or JSON string
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

    # gets any known keys returned by MCP 
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

    # finds report paths 
    for match in re.findall(r"(?:reports/[\w\-\./]+\.json|[\w\-\./]+/[^\s,;'\"<>()]+\.json|[^\s,;'\"<>()]+\.json)", text):
        clean_match = match.strip("'\"")
        if clean_match and clean_match not in report_paths:
            report_paths.append(clean_match)

    # finds checkpoint paths 
    for match in re.findall(r"(?:models/[\w\-\./]+\.(?:pth|pt)|[\w\-\./]+/[^\s,;'\"<>()]+\.(?:pth|pt)|[^\s,;'\"<>()]+\.(?:pth|pt))", text):
        clean_match = match.strip("'\"")
        if clean_match and clean_match not in checkpoint_paths:
            checkpoint_paths.append(clean_match)

    # finds verdict keywords
    if not verdicts:
        verdict_match = re.search(r"verdict[:\s=]+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        if verdict_match:
            verdicts.append(verdict_match.group(1))
        else:
            text_lower = text.lower()
            for v_word in ("attack", "clean", "poisoned", "unaffected", "suspicious", "repaired"):
                if re.search(rf"\b{v_word}\b", text_lower):
                    verdicts.append(v_word)
                    break

    return report_paths, checkpoint_paths, verdicts


def build_agent_run_summary(
    prompt: str,
    messages: list[Any],
    agent_name: str = "supervisor",
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    tool_messages_by_id: dict[str, Any] = {}
    ordered_tool_messages: list[Any] = []
    final_message = ""

    for msg in messages:
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            tool_messages_by_id[tool_call_id] = msg
        msg_type = getattr(msg, "type", None) or type(msg).__name__
        if msg_type in ("tool", "ToolMessage") or hasattr(msg, "tool_call_id"):
            ordered_tool_messages.append(msg)

        # tracks the final AI message
        if msg_type in ("ai", "AIMessage"):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content:
                final_message = content
            elif isinstance(content, list):
                final_message = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )

    steps: list[dict[str, Any]] = []
    tool_msg_index = 0

    # walk message history to find AI Messages
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls and hasattr(msg, "additional_kwargs"):
            tool_calls = msg.additional_kwargs.get("tool_calls", [])

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

            # matches the tool message
            tool_msg = None
            if tc_id and tc_id in tool_messages_by_id:
                tool_msg = tool_messages_by_id[tc_id]
            elif tool_msg_index < len(ordered_tool_messages):
                tool_msg = ordered_tool_messages[tool_msg_index]
            tool_msg_index += 1

            # extracts output paths and verdicts from tool result
            raw_result = getattr(tool_msg, "content", "") if tool_msg is not None else ""
            report_paths, checkpoint_paths, verdicts = _extract_paths_and_verdict(raw_result)

            step_data: dict[str, Any] = {
                "step": len(steps) + 1,
                "specialist": tc_name,
                "request": request_sent,
                "report_paths": report_paths,
                "report_path": report_paths[0] if report_paths else None,
                "checkpoint_paths": checkpoint_paths,
                "checkpoint_path": checkpoint_paths[0] if checkpoint_paths else None,
                "verdicts": verdicts,
                "verdict": verdicts[0] if verdicts else None,
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


async def supervisor_agent(
    prompt: str,
    model_name: str = "gpt-4o-mini",
    return_summary: bool = False,
):
    model = init_chat_model(model=model_name)
    agent = create_agent(model, [audit, detect, repair], system_prompt=SUPERVISOR_SYSTEM_PROMPT)
    result = await agent.ainvoke({"messages": prompt})
    messages = result.get("messages", [])
    final_message = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    if isinstance(final_message, list):
        final_message = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in final_message
        )
    if return_summary:
        summary = build_agent_run_summary(
            prompt=prompt,
            messages=messages,
            agent_name="supervisor",
            model_name=model_name,
        )
        return final_message, summary
    return final_message


if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Run a test suite (audit and detect) as needed on models/resnet18_poison.pth and run repairs if there is sufficient evidence of a backdoor."
    print(asyncio.run(supervisor_agent(prompt)))