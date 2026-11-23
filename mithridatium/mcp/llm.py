"""Chat model selection shared by all the agents."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_LLM = "gpt-4o-mini"


def resolve_llm(llm: str | None = None) -> str:
    """Pick the model name: explicit argument, then MITHRIDATIUM_LLM, then the default."""
    return llm or os.environ.get("MITHRIDATIUM_LLM") or DEFAULT_LLM


def chat_model(llm: str | None = None):
    # init_chat_model accepts "provider:model" (e.g. "ollama:llama3.1") or a bare
    # name it can infer a provider from (e.g. "gpt-4o-mini")
    return init_chat_model(model=resolve_llm(llm))
