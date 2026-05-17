"""
Retrieval worker: finds candidates matching skill/criteria queries.

Bound to the three search tools (skill, category, summary-level triage).
"""

from __future__ import annotations

from src.agents.react_worker import build_react_worker
from src.prompts.worker_prompts import RETRIEVAL_PROMPT
from src.retrieval.tools import (
    search_candidates_by_category,
    search_candidates_by_skill,
    search_summaries,
)

retrieval_node = build_react_worker(
    "retrieval",
    RETRIEVAL_PROMPT,
    [search_candidates_by_skill, search_candidates_by_category, search_summaries],
)
