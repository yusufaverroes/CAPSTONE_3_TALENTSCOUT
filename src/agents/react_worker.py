"""
Shared builder for the three worker nodes.

Each worker is a ``create_react_agent`` (its own ReAct tool-calling loop)
wrapped in a node that adapts it to ``AgentState``: it injects the active
sidebar category filter as context, runs the sub-agent, then extracts what
the parent graph needs — the final answer message, this turn's token cost,
and the structured source records the search tools attached as artifacts.

Only the final assistant message is returned to the parent ``messages``
channel; the sub-agent's intermediate tool chatter stays inside the worker.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src import config
from src.agents.state import AgentState


def build_react_worker(name: str, system_prompt: str, tools: list):
    """Return a LangGraph node for one worker agent."""
    model = ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=config.WORKER_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )
    agent = create_react_agent(model, tools, prompt=system_prompt)

    def _node(state: AgentState) -> dict:
        inputs = list(state["messages"])
        categories = state.get("category_filter") or []
        if categories:
            inputs = [
                SystemMessage(
                    content=(
                        f"Active sidebar category filter: {categories}. "
                        f"Pass the matching value as the `category` argument "
                        f"to search tools so results stay in scope."
                    )
                ),
                *inputs,
            ]

        result = agent.invoke({"messages": inputs})
        produced = result["messages"][len(inputs):]

        in_tok = out_tok = 0
        sources: list[dict] = []
        for msg in produced:
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                in_tok += usage.get("input_tokens", 0)
                out_tok += usage.get("output_tokens", 0)
            artifact = getattr(msg, "artifact", None)
            if isinstance(artifact, list):
                sources.extend(artifact)

        final = next(
            (m for m in reversed(produced) if isinstance(m, AIMessage) and m.content),
            AIMessage(content=""),
        )
        return {
            "messages": [final],
            "agent_path": [name],
            "token_usage": {"input_tokens": in_tok, "output_tokens": out_tok},
            "retrieved_sources": sources,
        }

    return _node
