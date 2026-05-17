"""
Cross-encoder reranking for retrieved candidates.

Initial vector retrieval (bi-encoder) is fast but coarse: the query and each
document are embedded independently. A cross-encoder re-scores every
(query, candidate) pair jointly for a sharper final ordering. The model is
loaded lazily on first use — importing this module must stay cheap, and the
first real call pays a one-time model download.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src import config

_MODEL: CrossEncoder | None = None


def _model() -> CrossEncoder:
    """One shared cross-encoder, built lazily.

    The first call downloads the model (~80MB) and is slow; every call after
    reuses the cached instance. Documented as a known first-call delay.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = CrossEncoder(config.RERANKER_MODEL)
    return _MODEL


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = config.RERANK_TOP_N,
) -> list[dict]:
    """Re-score source records against the query and return the best
    ``top_n``, highest first.

    Each candidate's ``snippet`` is the text scored — that is the retrieved
    text the record carries. The cross-encoder relevance score replaces the
    coarse retrieval ``score`` so the source-record schema stays stable and
    ``score`` always means "best available relevance". Input dicts are not
    mutated (a shallow copy is returned).
    """
    if not candidates:
        return []
    pairs = [(query, c["snippet"]) for c in candidates]
    scores = _model().predict(pairs)
    ranked = sorted(
        (dict(c, score=float(s)) for c, s in zip(candidates, scores)),
        key=lambda c: c["score"],
        reverse=True,
    )
    return ranked[:top_n]
