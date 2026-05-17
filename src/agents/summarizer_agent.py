"""
Summarizer worker: structured profile of one specific candidate.

Bound to the resume lookup tools; produces a fixed profile template.
"""

from __future__ import annotations

from src.agents.react_worker import build_react_worker
from src.prompts.worker_prompts import SUMMARIZER_PROMPT
from src.retrieval.tools import get_resume_full_text, get_resume_summary

summarizer_node = build_react_worker(
    "summarizer",
    SUMMARIZER_PROMPT,
    [get_resume_full_text, get_resume_summary],
)
