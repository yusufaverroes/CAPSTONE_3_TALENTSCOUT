"""
Multi-agent graph wiring.

Topology: the supervisor is the entry point. It routes to one worker; that
worker runs and control returns to the supervisor, which routes again or
finishes. A hard cap stops runaway routing loops: once the number of worker
hops taken (``len(agent_path)``) reaches ``config.MAX_SUPERVISOR_HOPS`` the
graph ends regardless of the supervisor's decision. Using the existing
``agent_path`` length as the counter avoids adding a field to the state.

Each worker is a tool-using ReAct agent. Workers never call each other; they
return to the supervisor, which decides whether to route again or finish.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src import config
from src.agents.comparison_agent import comparison_node
from src.agents.retrieval_agent import retrieval_node
from src.agents.state import AgentState
from src.agents.summarizer_agent import summarizer_node
from src.agents.supervisor import supervisor_node


def route_from_supervisor(state: AgentState) -> str:
    """Translate the supervisor decision into the next graph hop.

    Three end conditions, in order:
    1. The supervisor says FINISH.
    2. Scoped loop-breaker: the supervisor routes to the *same* worker that
       just ran. No new user input can arrive mid-run, so re-running the
       worker that already handled this turn is always redundant — treat the
       request as done. A legitimate multi-step chain routes to a *different*
       worker (e.g. retrieval -> comparison), so this never blocks it.
    3. The worker-hop hard cap (last-resort safety net).
    """
    nxt = state["next_agent"]
    if nxt == "FINISH":
        return END
    path = state["agent_path"]
    if path and nxt == path[-1]:
        return END
    if len(path) >= config.MAX_SUPERVISOR_HOPS:
        return END
    return nxt


def build_graph():
    """Wire the supervisor + workers into a compiled graph."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("comparison", comparison_node)
    builder.add_node("summarizer", summarizer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retrieval": "retrieval",
            "comparison": "comparison",
            "summarizer": "summarizer",
            END: END,
        },
    )
    builder.add_edge("retrieval", "supervisor")
    builder.add_edge("comparison", "supervisor")
    builder.add_edge("summarizer", "supervisor")

    return builder.compile()


app_graph = build_graph()
