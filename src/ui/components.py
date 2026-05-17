"""
Rendering helpers for the chat UI (SPEC §5.8).

These own *all* presentation: ``app.py`` decides what to show, these decide how
it looks. Every HTML string uses only the ``ts-`` classes from ``styles.py``;
data interpolated into HTML is escaped because resume text is arbitrary.

Two presentation rules worth calling out:

* Reranker scores arrive as raw cross-encoder logits (unbounded, often
  negative). They are squashed through a sigmoid to a 0–1 pseudo-probability
  for display — monotonic, so the ranking order the model saw is preserved,
  and not distorted across candidates the way a softmax-over-set would be for
  a 5-item list (D-5.1).
* Source records are heterogeneous: search tools attach a snippet and a
  score; fetch-by-id lookups attach only resume_id + category (snippet/score
  are None). The panel renders each field only when present, so a grounded
  comparison/summarizer answer still shows *which* candidate it used.
"""

from __future__ import annotations

import html
import math

import streamlit as st

_PILL_CLASS = {
    "retrieval": "ts-pill--retrieval",
    "comparison": "ts-pill--comparison",
    "summarizer": "ts-pill--summarizer",
}


def _sigmoid(logit: float) -> float:
    """Numerically stable logistic squash of a cross-encoder logit to 0–1."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    e = math.exp(logit)
    return e / (1.0 + e)


def render_source_panel(sources: list[dict]) -> None:
    """The tinted-blue panel of retrieved document references.

    Renders adaptively over the heterogeneous source schema: score and
    snippet appear only when the record carries them (search hits), so
    lookup-only records still attribute the answer to a candidate.
    """
    if not sources:
        return
    rows = []
    for s in sources:
        cat = html.escape(str(s.get("category", "")))
        section = html.escape(str(s.get("section", "")))
        rid = html.escape(str(s.get("resume_id", "?")))
        score = s.get("score")
        score_html = (
            f"<span class='ts-source-score'>{_sigmoid(float(score)):.0%}</span>"
            if score is not None
            else ""
        )
        snippet = s.get("snippet")
        snippet_html = (
            f"<div class='ts-source-snippet'>{html.escape(str(snippet))}</div>"
            if snippet
            else ""
        )
        rows.append(
            f"<div class='ts-source-item'>{score_html}"
            f"<span class='ts-source-id'>resume #{rid}</span>"
            f"<span class='ts-source-tag'>{cat}</span>"
            f"<span class='ts-source-tag'>{section}</span>"
            f"{snippet_html}</div>"
        )
    st.markdown(
        f"<div class='ts-source-panel'>"
        f"<div class='ts-source-head'>Sumber ({len(sources)})</div>"
        f"{''.join(rows)}</div>",
        unsafe_allow_html=True,
    )


def render_agent_path_expander(agent_path: list[str]) -> None:
    """Expander showing the routing trace for one answer."""
    if not agent_path:
        return
    trace = " → ".join(["supervisor", *agent_path])
    with st.expander(f"Jalur agent: {trace}"):
        st.write(
            "Supervisor merouting tiap hop; worker yang menangani query ini "
            "tercatat berurutan."
        )


def render_metadata_badges(
    token_count: dict, cost_idr: float, history_length: int
) -> None:
    """The inline badge row below an agent response."""
    total = token_count.get("input_tokens", 0) + token_count.get("output_tokens", 0)
    badges = [
        f"🔢 {total} token "
        f"(in {token_count.get('input_tokens', 0)} / "
        f"out {token_count.get('output_tokens', 0)})",
        f"💰 Rp {cost_idr:,.2f}",
        f"💬 {history_length} turn",
    ]
    st.markdown(
        "<div class='ts-meta'>"
        + "".join(f"<span class='ts-meta-badge'>{b}</span>" for b in badges)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_chat_message(
    role: str,
    content: str,
    agent_name: str | None = None,
    sources: list[dict] | None = None,
    metadata: dict | None = None,
) -> None:
    """Render any single chat message — the one entry point app.py replays.

    User turns are plain. Assistant turns get an agent identity pill, the
    markdown body, the source panel, then the routing expander + metadata
    badge row (the order the SPEC §6.3 mockup specifies).
    """
    with st.chat_message(role):
        if role == "assistant" and agent_name:
            pill = _PILL_CLASS.get(agent_name, "ts-pill--default")
            st.markdown(
                f"<span class='ts-pill {pill}'>via {agent_name} agent</span>",
                unsafe_allow_html=True,
            )
        st.markdown(content)
        if sources:
            render_source_panel(sources)
        if metadata:
            render_agent_path_expander(metadata.get("agent_path", []))
            render_metadata_badges(
                metadata.get("token_usage", {}),
                metadata.get("cost_idr", 0.0),
                metadata.get("history_length", 0),
            )


def render_session_stats(
    token_count: dict, cost_idr: float, query_count: int
) -> None:
    """The cumulative 'Sesi ini' block in the sidebar."""
    total = token_count.get("input_tokens", 0) + token_count.get("output_tokens", 0)
    st.metric("Query", query_count)
    st.metric("Token kumulatif", f"{total:,}")
    st.metric("Biaya kumulatif", f"Rp {cost_idr:,.2f}")
