"""
LangChain tools over the resume vector store.

Each tool is module-level and decorated with ``@tool`` so it can be bound to
any agent. Docstrings are written for the LLM: they are read at routing time
to decide which tool fits the user's request, so they state precisely what
the tool does and what it returns. User-facing text is translated to Bahasa
Indonesia later by the agent prompt, not here.

The three search tools use ``response_format="content_and_artifact"``: the
model reads a compact human-readable string (the *content*), while the
structured source records (the *artifact*) ride along on the ToolMessage so
the graph can accumulate them into ``retrieved_sources`` for the UI source
panel without re-parsing text.

Chunk search runs a deterministic two-stage pipeline — bi-encoder retrieve
(K, coarse) then cross-encoder rerank to the best one chunk per candidate
(top-N). Reranking is a fixed pipeline step, not an LLM-invoked tool: the
model only ever sees the *content* string, so it could not faithfully pass a
structured candidate list back into a rerank tool. Summary search is coarse
candidate-level triage and is intentionally not reranked.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src import config
from src.retrieval.reranker import rerank
from src.retrieval.vector_store import TalentScoutVectorStore

_STORE: TalentScoutVectorStore | None = None


def _store() -> TalentScoutVectorStore:
    """One shared store, built lazily.

    Importing this module must not require a live Qdrant connection, otherwise
    the agent graph and the tests could not be imported offline. The store is
    constructed on first tool use and reused thereafter.
    """
    global _STORE
    if _STORE is None:
        _STORE = TalentScoutVectorStore()
    return _STORE


def _rerank_dedup(query: str, records: list[dict]) -> list[dict]:
    """Cross-encoder rerank the full pool, then keep the best chunk per
    candidate.

    The reranker scores every retrieved chunk, so one resume can hold several
    chunks in the pool. We rerank all of them, then walk the reranked order
    keeping the first (highest-scoring) chunk per ``resume_id`` and return the
    top ``RERANK_TOP_N`` distinct candidates — a recruitment shortlist wants
    distinct people, not the same person repeated.
    """
    ranked = rerank(query, records, top_n=len(records))
    seen: set = set()
    unique: list[dict] = []
    for r in ranked:
        if r["resume_id"] in seen:
            continue
        seen.add(r["resume_id"])
        unique.append(r)
    return unique[: config.RERANK_TOP_N]


def _format(records: list[dict]) -> str:
    """Render source records into a compact list the model can reason over."""
    if not records:
        return "No matching candidates found."
    lines = []
    for r in records:
        lines.append(
            f"- resume_id={r['resume_id']} | category={r['category']} | "
            f"section={r['section']} | score={r['score']}\n  {r['snippet']}"
        )
    return "\n".join(lines)


@tool(response_format="content_and_artifact")
def search_candidates_by_skill(
    skill_query: str, category: str | None = None
) -> tuple[str, list[dict]]:
    """Find candidates whose resume detail matches a specific skill or
    technology (e.g. "AWS", "financial modelling"). Searches resume chunks,
    floats Skills-section matches up, then cross-encoder reranks to the best
    distinct candidates. Optionally restrict to one job category such as
    "INFORMATION-TECHNOLOGY". Returns a readable ranked shortlist; structured
    source records {resume_id, category, section, snippet, score} are attached
    as artifact.
    """
    records = _store().search_chunks(
        skill_query, category_filter=category, section_filter="skills"
    )
    records = _rerank_dedup(skill_query, records)
    return _format(records), records


@tool(response_format="content_and_artifact")
def search_candidates_by_category(
    query: str, category: str
) -> tuple[str, list[dict]]:
    """Semantic search for candidates within a single known job category. Use
    when the request is scoped to a category (e.g. "marketing managers in
    SALES"). Chunks are cross-encoder reranked to the best distinct
    candidates. Returns a readable ranked shortlist; structured source records
    {resume_id, category, section, snippet, score} are attached as artifact.
    """
    records = _store().search_chunks(query, category_filter=category)
    records = _rerank_dedup(query, records)
    return _format(records), records


@tool(response_format="content_and_artifact")
def search_summaries(
    query: str, category: str | None = None
) -> tuple[str, list[dict]]:
    """Fast candidate-level triage: one holistic summary per candidate, ideal
    for "which candidates fit X" before drilling into detail. Optionally
    restrict to one job category. Returns a readable shortlist; structured
    source records {resume_id, category, section, snippet, score} are attached
    as artifact.
    """
    records = _store().search_summaries(query, category_filter=category)
    return _format(records), records


def _lookup_source(resume_id: int, category: str, label: str) -> dict:
    """Minimal source record for a fetch-by-id lookup.

    A lookup answers from a whole resume (or its one-line summary), not from
    ranked chunks, so there is no snippet or rerank score — those stay None
    and the panel renders the remaining fields adaptively. The point is the
    source panel still shows *which* candidate the answer is grounded in,
    keeping the grounding evidence visible for comparison/summarizer turns.
    """
    return {
        "resume_id": resume_id,
        "category": category,
        "section": label,
        "snippet": None,
        "score": None,
    }


@tool(response_format="content_and_artifact")
def get_resume_full_text(resume_id: int) -> tuple[str, list[dict]]:
    """Fetch the full resume of one candidate by numeric ID, with every chunk
    reassembled in order. Use for deep comparison or detailed summarisation of
    a specific candidate already identified by a prior search. Raises if the
    ID is unknown. A minimal source record {resume_id, category} is attached
    as artifact so the answer stays attributable in the UI.
    """
    full = _store().get_full_resume(resume_id)
    content = (
        f"[resume_id={full['resume_id']} | category={full['category']} | "
        f"{full['n_chunks']} chunks]\n\n{full['full_text']}"
    )
    return content, [_lookup_source(full["resume_id"], full["category"], "full resume")]


@tool(response_format="content_and_artifact")
def get_resume_summary(resume_id: int) -> tuple[str, list[dict]]:
    """Fetch the one-line LLM-generated profile summary of a candidate by
    numeric ID. Use for a quick overview without pulling the full resume.
    Raises if the ID is unknown. A minimal source record {resume_id, category}
    is attached as artifact so the answer stays attributable in the UI.
    """
    rec = _store().get_summary(resume_id)
    return rec["summary"], [
        _lookup_source(rec["resume_id"], rec["category"], "profile summary")
    ]
