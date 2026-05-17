"""
Comparison worker: side-by-side judgement of named candidates.

Bound to the resume lookup tools; produces a Markdown comparison table.
"""

from __future__ import annotations

from src.agents.react_worker import build_react_worker
from src.prompts.worker_prompts import COMPARISON_PROMPT
from src.retrieval.tools import get_resume_full_text, get_resume_summary

comparison_node = build_react_worker(
    "comparison",
    COMPARISON_PROMPT,
    [get_resume_full_text, get_resume_summary],
)
