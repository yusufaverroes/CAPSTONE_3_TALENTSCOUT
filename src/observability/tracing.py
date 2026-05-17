"""
Langfuse tracing for the agent runtime.

One LangChain ``CallbackHandler`` attached to ``app_graph.invoke`` traces the
whole multi-agent run — every LLM call (supervisor + each worker) and every
tool call (search / rerank / lookup) nested automatically. This is purely
observability for debugging and the demo; per-query token totals shown in the
UI still come from the manual accumulator in the agent state, not from here.

Tracing is optional and must never break the app: if Langfuse credentials are
absent the callbacks list is empty and the graph runs untraced. Client init
and the handler are built lazily so importing this module stays cheap.
"""

from __future__ import annotations

from src import config

_CLIENT = None


def _enabled() -> bool:
    return bool(config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY)


def _client():
    """Configure the Langfuse singleton once, from central config."""
    global _CLIENT
    if _CLIENT is None:
        from langfuse import Langfuse, get_client

        Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        _CLIENT = get_client()
    return _CLIENT


def get_tracing_callbacks() -> list:
    """Callbacks to spread into ``app_graph.invoke(config={"callbacks": ...})``.

    Empty list when credentials are missing — the caller passes it
    unconditionally and the graph simply runs untraced.
    """
    if not _enabled():
        return []
    from langfuse.langchain import CallbackHandler

    _client()
    return [CallbackHandler()]


def flush() -> None:
    """Flush buffered traces — call once at the end of a short-lived run."""
    if _enabled():
        _client().flush()
