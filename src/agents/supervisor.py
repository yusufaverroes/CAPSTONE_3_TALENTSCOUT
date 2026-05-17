"""
Supervisor node: the LangGraph entry point that decides routing.

It calls the chat model with the supervisor prompt and the conversation,
constraining the answer to a structured routing decision. It emits only the
routing target and the tokens its own call cost — it never appends to
``agent_path`` (that channel records workers, so its length doubles as the
worker-hop counter the graph uses for the hard cap) and never produces
user-facing text.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src import config
from src.agents.state import AgentState
from src.prompts.supervisor_prompts import SUPERVISOR_SYSTEM_PROMPT


class RoutingDecision(BaseModel):
    """Structured output the supervisor is forced to return.

    ``request_satisfied`` is declared first so the model commits to the
    satisfaction judgement before choosing a route. When it is true the node
    overrides ``next_agent`` to FINISH, so an already-answered request can
    never re-route into a worker loop.
    """

    request_satisfied: bool = Field(
        description=(
            "True if the last message already fulfils the user's most recent "
            "request and nothing new was asked after it. Decide this FIRST."
        )
    )
    next_agent: Literal["retrieval", "comparison", "summarizer", "FINISH"] = Field(
        description=(
            "Worker for the outstanding work. MUST be FINISH if "
            "request_satisfied is true."
        )
    )


_LLM = None


def _llm():
    """Structured-output chat model, built once.

    ``include_raw=True`` keeps the underlying AIMessage so we can read its
    ``usage_metadata`` and accumulate tokens manually. Construction is offline;
    only ``.invoke`` reaches the network.
    """
    global _LLM
    if _LLM is None:
        base = ChatOpenAI(
            model=config.CHAT_MODEL,
            temperature=config.SUPERVISOR_TEMPERATURE,
            api_key=config.OPENAI_API_KEY,
        )
        _LLM = base.with_structured_output(RoutingDecision, include_raw=True)
    return _LLM


def supervisor_node(state: AgentState) -> dict:
    """Pick the next worker (or FINISH) and report this call's token cost."""
    result = _llm().invoke(
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), *state["messages"]]
    )
    decision: RoutingDecision = result["parsed"]
    # Satisfaction is enforced, not merely suggested: if the model judged the
    # request already answered, terminate regardless of the worker it named.
    next_agent = "FINISH" if decision.request_satisfied else decision.next_agent
    usage = getattr(result["raw"], "usage_metadata", None) or {}
    return {
        "next_agent": next_agent,
        "token_usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }
