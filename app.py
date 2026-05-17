"""
TalentScout AI — Streamlit entry point.

Thin orchestration: owns session state and the request lifecycle, delegates
every pixel to ``src.ui.components`` and every style to ``src.ui.styles``
(SPEC §5.11). The graph is invoked with ``stream`` so a step-by-step status
shows the supervisor and workers acting in turn; Langfuse callbacks ride the
same invocation and are flushed once the run ends.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src import config
from src.agents.graph import app_graph
from src.observability import tracing
from src.observability.cost import tokens_to_idr
from src.ui import components
from src.ui.styles import INLINE_CSS

QUICK_PROMPTS = [
    "Cari kandidat IT cloud",
    "Bandingkan 2 kandidat HR",
    "Ringkas profil ID 12345",
]
_STEP_LABEL = {
    "retrieval": "Mencari kandidat…",
    "comparison": "Membandingkan kandidat…",
    "summarizer": "Menyusun ringkasan…",
}

st.set_page_config(page_title="TalentScout AI", layout="wide", page_icon="🧭")
st.markdown(f"<style>{INLINE_CSS}</style>", unsafe_allow_html=True)


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("cum_tokens", {"input_tokens": 0, "output_tokens": 0})
    ss.setdefault("cum_cost", 0.0)
    ss.setdefault("query_count", 0)


def _history_messages() -> list:
    """Recent turns as LangChain messages (D-5.2: send last N turns)."""
    recent = st.session_state.messages[-config.CHAT_HISTORY_TURNS_TO_SEND * 2 :]
    return [
        (HumanMessage if m["role"] == "user" else AIMessage)(content=m["content"])
        for m in recent
    ]


def _final_answer(messages: list) -> str:
    """Last assistant message with text — robust if the graph finished
    without a worker turn (then messages[-1] is the user's own prompt)."""
    for m in reversed(messages or []):
        if isinstance(m, AIMessage) and m.content:
            return m.content
    return ""


def _dedup_sources(sources: list[dict]) -> list[dict]:
    seen, out = set(), []
    for s in sources or []:
        key = (s.get("resume_id"), s.get("section"), s.get("snippet"))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _run_query(prompt: str, categories: list[str]) -> dict:
    """Stream the graph, surfacing each hop as a status step; return the
    assistant message dict to append to history."""
    state = {
        "messages": [*_history_messages(), HumanMessage(content=prompt)],
        "next_agent": "",
        "agent_path": [],
        "retrieved_sources": [],
        "category_filter": categories,
        "token_usage": {},
    }
    cfg = {"callbacks": tracing.get_tracing_callbacks(), "recursion_limit": 25}
    final = None
    with st.status("Merouting permintaan…", expanded=False) as status:
        try:
            for snap in app_graph.stream(state, config=cfg, stream_mode="values"):
                final = snap
                path = snap.get("agent_path") or []
                status.update(
                    label=_STEP_LABEL.get(path[-1], "Memproses…")
                    if path
                    else "Merouting permintaan…"
                )
            status.update(label="Selesai", state="complete")
        except Exception as exc:  # surfaced, not swallowed (CLAUDE.md §7)
            status.update(label="Gagal", state="error")
            return {
                "role": "assistant",
                "content": f"⚠️ **Maaf, terjadi error.**\n\n```\n{exc}\n```",
                "agent_name": None,
                "sources": [],
                "metadata": None,
            }
        finally:
            tracing.flush()

    path = final.get("agent_path") or []
    tok = final.get("token_usage") or {"input_tokens": 0, "output_tokens": 0}
    cost = tokens_to_idr(tok.get("input_tokens", 0), tok.get("output_tokens", 0))
    ss = st.session_state
    ss.cum_tokens["input_tokens"] += tok.get("input_tokens", 0)
    ss.cum_tokens["output_tokens"] += tok.get("output_tokens", 0)
    ss.cum_cost += cost
    ss.query_count += 1
    return {
        "role": "assistant",
        "content": _final_answer(final.get("messages")) or "_(tidak ada jawaban)_",
        "agent_name": path[-1] if path else None,
        "sources": _dedup_sources(final.get("retrieved_sources")),
        "metadata": {
            "agent_path": path,
            "token_usage": tok,
            "cost_idr": cost,
            "history_length": len(ss.messages) + 2,
        },
    }


_init_state()

with st.sidebar:
    st.subheader("Filter kategori")
    categories = st.multiselect(
        "Batasi ke kategori", config.CATEGORIES, default=[],
        label_visibility="collapsed",
    )
    st.subheader("Sesi ini")
    components.render_session_stats(
        st.session_state.cum_tokens,
        st.session_state.cum_cost,
        st.session_state.query_count,
    )
    st.subheader("Tautan")
    st.markdown(f"[Langfuse traces]({config.LANGFUSE_HOST})")
    if config.GITHUB_URL:
        st.markdown(f"[GitHub repo]({config.GITHUB_URL})")
    if st.button("Reset percakapan", use_container_width=True):
        st.session_state.messages = []
        st.session_state.cum_tokens = {"input_tokens": 0, "output_tokens": 0}
        st.session_state.cum_cost = 0.0
        st.session_state.query_count = 0
        st.rerun()

st.markdown(
    "<div class='ts-header'>"
    "<span class='ts-header-title'>🧭 TalentScout AI</span>"
    "<span class='ts-header-badge'>2403 CV terindeks</span>"
    "<span class='ts-header-badge'>Multi-agent</span></div>",
    unsafe_allow_html=True,
)

picked = None
for col, text in zip(st.columns(len(QUICK_PROMPTS)), QUICK_PROMPTS):
    if col.button(text, use_container_width=True):
        picked = text

for m in st.session_state.messages:
    components.render_chat_message(
        m["role"], m["content"], m.get("agent_name"),
        m.get("sources"), m.get("metadata"),
    )

prompt = st.chat_input("Tanyakan tentang kandidat...") or picked
if prompt:
    components.render_chat_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer = _run_query(prompt, categories)
    st.session_state.messages.append(answer)
    components.render_chat_message(
        answer["role"], answer["content"], answer.get("agent_name"),
        answer.get("sources"), answer.get("metadata"),
    )
