"""
Shared state for the multi-agent graph.

The supervisor and the three worker nodes all read from and write to one
``AgentState``. Fields that several nodes contribute to over a single query
(the message log, the agent trail, retrieved sources, token totals) carry a
reducer so sequential writes across routing hops accumulate instead of
overwrite. Fields the supervisor simply replaces each hop (the routing
decision) and fields injected once at graph entry (the sidebar category
filter) are plain values with no reducer.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def _accumulate_tokens(left: dict, right: dict) -> dict:
    """Sum input/output token counts across every LLM call in one query.

    Robust to a missing/empty side: each key defaults to 0, so seeding the
    entry state with ``{}`` (or omitting it) still works.
    """
    return {
        "input_tokens": left.get("input_tokens", 0) + right.get("input_tokens", 0),
        "output_tokens": left.get("output_tokens", 0)
        + right.get("output_tokens", 0),
    }


class AgentState(TypedDict):
    """State channel contract for the supervisor + worker graph.

    Graph entry must seed the accumulator channels (``agent_path`` and
    ``retrieved_sources`` as ``[]``, ``token_usage`` as ``{}``) so the
    reducers have a left operand on the first write.
    """

    # Full conversation. add_messages appends new turns and de-dups by id,
    # so a node returning only its new message does not clobber history.
    messages: Annotated[list[BaseMessage], add_messages]

    # Supervisor's routing target for the next hop: "retrieval",
    # "comparison", "summarizer", or "FINISH". Replaced every hop.
    next_agent: str

    # Every agent that handled this query, in order, for UI visibility. Each
    # worker returns its own name as a one-element list; operator.add
    # concatenates them across hops.
    agent_path: Annotated[list[str], operator.add]

    # Source records backing the answer, accumulated across worker calls:
    # {resume_id, category, section, snippet, score}. Rendered in the
    # source-attribution panel.
    retrieved_sources: Annotated[list[dict], operator.add]

    # Categories selected in the sidebar, injected once at graph entry.
    # Empty list means no category filter (search the whole corpus).
    category_filter: list[str]

    # Running {"input_tokens", "output_tokens"} totals for the current query,
    # summed across every LLM call so the UI can show per-query token + cost.
    token_usage: Annotated[dict, _accumulate_tokens]
