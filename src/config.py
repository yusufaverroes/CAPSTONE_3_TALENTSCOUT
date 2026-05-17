"""
Centralized configuration for TalentScout AI.

All credentials and tunable parameters live here. Modules import from this
module rather than reading env vars or st.secrets directly.

Secrets resolution order:
    1. Environment variable (covers local dev with .env via python-dotenv).
    2. Fallback to st.secrets when running under Streamlit Cloud.
    3. Raise a clear error if neither source provides the value.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve a secret from env vars first, then st.secrets, else default."""
    value = os.environ.get(key)
    if value:
        return value

    try:
        import streamlit as st

        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except (ImportError, FileNotFoundError, Exception):
        pass

    return default


def _require_secret(key: str) -> str:
    """Like _get_secret but raises if missing — for non-optional credentials."""
    value = _get_secret(key)
    if not value:
        raise RuntimeError(
            f"Missing required secret: {key}. "
            f"Set it in .env, .streamlit/secrets.toml, or your environment."
        )
    return value


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = _require_secret("OPENAI_API_KEY")

QDRANT_URL: str = _require_secret("QDRANT_URL")
QDRANT_API_KEY: str = _require_secret("QDRANT_API_KEY")

LANGFUSE_PUBLIC_KEY: Optional[str] = _get_secret("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY: Optional[str] = _get_secret("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST: str = (
    _get_secret("LANGFUSE_HOST", "https://cloud.langfuse.com")
    or "https://cloud.langfuse.com"
)

# Public GitHub repo URL, surfaced as a sidebar link (SPEC §6.2). Optional:
# the link is simply omitted when unset so local runs don't show a dead link.
GITHUB_URL: Optional[str] = _get_secret("GITHUB_URL")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = "text-embedding-3-small"   # 1536 dim, ~5x cheaper than -large
CHAT_MODEL: str = "gpt-4o-mini"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

CHUNK_COLLECTION: str = "resume_chunks"
SUMMARY_COLLECTION: str = "resume_summaries"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE_FALLBACK: int = 500          # chars, used when section detection fails
CHUNK_OVERLAP_FALLBACK: int = 50
EMBEDDING_BATCH_SIZE: int = 50          # chunks per OpenAI embedding API call


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K_CHUNK_SEARCH: int = 15            # initial vector search
TOP_K_SUMMARY_SEARCH: int = 5
RERANK_TOP_N: int = 5                   # results returned after cross-encoder rerank
MMR_LAMBDA: float = 0.5                 # 0 = max diversity, 1 = pure relevance
MMR_FETCH_K: int = 60                   # candidate pool MMR diversifies from (>> k)


# ---------------------------------------------------------------------------
# Agent runtime
# ---------------------------------------------------------------------------

MAX_SUPERVISOR_HOPS: int = 5            # hard cap to prevent routing loops
SUPERVISOR_TEMPERATURE: float = 0.0     # deterministic routing decisions
WORKER_TEMPERATURE: float = 0.3         # mild variation for natural responses


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

CHAT_HISTORY_TURNS_TO_SEND: int = 10    # most recent N turns sent as context


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# gpt-4o-mini list price snapshot 2026-05.
INPUT_TOKEN_PRICE_USD_PER_1M: float = 0.15
OUTPUT_TOKEN_PRICE_USD_PER_1M: float = 0.60
USD_TO_IDR: float = 17000.0


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

# Stored in UPPERCASE to match the raw values in Resume.csv (and therefore the
# value used as a Qdrant metadata filter).
CATEGORIES: list[str] = [
    "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE",
    "AVIATION", "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF",
    "CONSTRUCTION", "CONSULTANT", "DESIGNER", "DIGITAL-MEDIA", "ENGINEERING",
    "FINANCE", "FITNESS", "HEALTHCARE", "HR", "INFORMATION-TECHNOLOGY",
    "PUBLIC-RELATIONS", "SALES", "TEACHER",
]
