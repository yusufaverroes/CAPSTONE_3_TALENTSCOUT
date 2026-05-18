"""
Supervisor node: the LangGraph entry point that decides routing.

It calls the chat model with the supervisor prompt and the conversation,
constraining the answer to a structured routing decision. It emits the
routing target and the tokens its own call cost, and it never appends to
``agent_path`` (that channel records workers, so its length doubles as the
worker-hop counter the graph uses for the hard cap).

It produces user-facing text in exactly one case: an out-of-scope request.
The scope gate is decided here (not in a worker) because the supervisor is the
only neutral step — a worker is already instructed to find candidates and will
answer anyway. On an out-of-scope request the node appends a fixed Bahasa
Indonesia refusal and routes to FINISH, so the turn ends before any worker
runs.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src import config
from src.agents.state import AgentState
from src.prompts.supervisor_prompts import SUPERVISOR_SYSTEM_PROMPT

# Fixed refusal for out-of-scope requests. Deterministic (no extra LLM call):
# the scope judgement is already made, only a canned reply is needed.
OUT_OF_SCOPE_REPLY = (
    "Maaf, saya asisten rekrutmen TalentScout dan hanya bisa membantu seputar "
    "kandidat dalam basis data CV — misalnya mencari, membandingkan, atau "
    "meringkas profil kandidat. Pertanyaan tadi di luar lingkup itu. Silakan "
    "ajukan pertanyaan terkait kandidat."
)


class RoutingDecision(BaseModel):
    """Structured output the supervisor is forced to return.

    Field order is the decision order the model must follow. ``in_scope`` is
    declared first so the scope judgement is committed before anything else;
    when it is false the node refuses and the other two fields are ignored.
    ``request_satisfied`` is next so the model commits to the satisfaction
    judgement before choosing a route — when it is true the node overrides
    ``next_agent`` to FINISH, so an already-answered request can never
    re-route into a worker loop.
    """

    in_scope: bool = Field(
        description=(
            "True if the LATEST user message is about candidates in the "
            "resume corpus (finding/comparing/summarising them or asking "
            "about their skills, experience, fit), OR an elliptical follow-up "
            "that resolves to such a request. False for any off-topic request "
            "(general knowledge, news, politics, chit-chat). Judge the latest "
            "message ALONE — earlier recruitment turns do not make it in "
            "scope. Decide this FIRST."
        )
    )
    request_satisfied: bool = Field(
        description=(
            "True if the last message already fulfils the user's most recent "
            "request and nothing new was asked after it. Decide this AFTER "
            "in_scope (only matters when in_scope is true)."
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
    usage = getattr(result["raw"], "usage_metadata", None) or {}
    token_usage = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }

    # Scope gate runs first: an out-of-scope request is refused here and the
    # turn ends via FINISH, so it can never reach a worker that would answer
    # it anyway. The refusal is appended to ``messages`` so the UI's
    # last-AIMessage extraction surfaces it as the turn's answer.
    if not decision.in_scope:
        return {
            "next_agent": "FINISH",
            "messages": [AIMessage(content=OUT_OF_SCOPE_REPLY)],
            "token_usage": token_usage,
        }

    # Satisfaction is enforced, not merely suggested: if the model judged the
    # request already answered, terminate regardless of the worker it named.
    next_agent = "FINISH" if decision.request_satisfied else decision.next_agent
    return {
        "next_agent": next_agent,
        "token_usage": token_usage,
    }
