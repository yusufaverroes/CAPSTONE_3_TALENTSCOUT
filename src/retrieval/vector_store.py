"""
Vector store access layer for TalentScout.

Wraps the two Qdrant collections behind one object so callers never have to
know which collection answers a question:

    - resume_summaries : one point per candidate (holistic, deduplicated) —
      used for coarse candidate-level triage.
    - resume_chunks    : many points per candidate — used to zoom into the
      detail of a shortlisted candidate.

Payload layout is LangChain-compatible: ``page_content`` holds the embedded
text and ``metadata`` holds ``{resume_id, category, section, chunk_index}``.
Every metadata filter therefore addresses the nested key ``metadata.<field>``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src import config


class TalentScoutVectorStore:
    """Single entry point for all read access to the resume vector store."""

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,  # Qdrant Cloud can cold-start for minutes on first call
        )
        # The query must land in the same vector space as the documents it is
        # compared against, so the embedding model must match ingestion exactly.
        self._embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
        # Defaults (content_payload_key='page_content',
        # metadata_payload_key='metadata', distance=COSINE) already match the
        # payload written at ingestion, so no overrides are needed here.
        self._summaries = QdrantVectorStore(
            client=self._client,
            collection_name=config.SUMMARY_COLLECTION,
            embedding=self._embeddings,
        )
        self._chunks = QdrantVectorStore(
            client=self._client,
            collection_name=config.CHUNK_COLLECTION,
            embedding=self._embeddings,
        )

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _category_filter(category: str | None) -> models.Filter | None:
        """Build a pre-filter on ``metadata.category``, or None for no filter.

        Category values are stored UPPERCASE (raw Resume.csv values), so the
        input is normalised before matching. An unknown category raises rather
        than silently returning zero hits: a wrong filter that yields an empty
        result set is indistinguishable from "no matching candidates" unless we
        fail loud here.
        """
        if not category:
            return None
        normalised = category.strip().upper()
        if normalised not in config.CATEGORIES:
            raise ValueError(
                f"Unknown category {category!r}. "
                f"Expected one of {config.CATEGORIES}."
            )
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.category",
                    match=models.MatchValue(value=normalised),
                )
            ]
        )

    @staticmethod
    def _to_source(doc: Document, score: float | None) -> dict:
        """Flatten a retrieved Document into the source-record contract.

        Summary points carry no ``section`` in their payload, so they fall
        back to the literal ``"summary"`` for a consistent source panel.
        """
        meta = doc.metadata
        return {
            "resume_id": meta["resume_id"],
            "category": meta["category"],
            "section": meta.get("section", "summary"),
            "snippet": doc.page_content[:300],
            "score": score,
        }

    # ----- search ----------------------------------------------------------

    def search_summaries(
        self,
        query: str,
        k: int = config.TOP_K_SUMMARY_SEARCH,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Coarse candidate-level triage over one-point-per-candidate summaries.

        Pure similarity: summaries are 1:1 with candidates and already
        deduplicated, so there is no inter-result redundancy for MMR to fix.
        """
        hits = self._summaries.similarity_search_with_score(
            query,
            k=k,
            filter=self._category_filter(category_filter),
        )
        return [self._to_source(doc, score) for doc, score in hits]

    def search_chunks(
        self,
        query: str,
        k: int = config.TOP_K_CHUNK_SEARCH,
        category_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[dict]:
        """Fine-grained search over many-chunks-per-candidate detail.

        Uses MMR (a candidate's chunks are highly redundant, so pure
        similarity would return five fragments of one person). ``section_filter``
        is a SOFT preference: matching sections are floated to the top but
        nothing is ever dropped, so a section hint can never cause a false
        negative.
        """
        query_vector = self._embeddings.embed_query(query)
        hits = self._chunks.max_marginal_relevance_search_with_score_by_vector(
            embedding=query_vector,
            k=k,
            fetch_k=config.MMR_FETCH_K,
            lambda_mult=config.MMR_LAMBDA,
            filter=self._category_filter(category_filter),
        )
        if section_filter:
            wanted = section_filter.strip().lower()
            hits = sorted(
                hits,
                key=lambda h: h[0].metadata.get("section", "").lower() != wanted,
            )
        return [self._to_source(doc, score) for doc, score in hits]

    # ----- direct fetch by resume_id --------------------------------------

    def _scroll_by_resume_id(self, collection: str, resume_id) -> list:
        """Every payload for one resume_id, paginated so nothing is truncated.

        The payload stores ``resume_id`` as the numeric CSV id, so the input
        is coerced to int — a string id would match zero points silently.
        """
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.resume_id",
                    match=models.MatchValue(value=int(resume_id)),
                )
            ]
        )
        records: list = []
        offset = None
        while True:
            batch, offset = self._client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=offset,
            )
            records.extend(batch)
            if offset is None:
                break
        return records

    def get_full_resume(self, resume_id) -> dict:
        """Reassemble every chunk of one resume, ordered by chunk_index."""
        records = self._scroll_by_resume_id(config.CHUNK_COLLECTION, resume_id)
        if not records:
            raise KeyError(f"No chunks found for resume_id {resume_id!r}.")
        chunks = sorted(
            (r.payload for r in records),
            key=lambda p: p["metadata"]["chunk_index"],
        )
        return {
            "resume_id": chunks[0]["metadata"]["resume_id"],
            "category": chunks[0]["metadata"]["category"],
            "full_text": "\n\n".join(c["page_content"] for c in chunks),
            "n_chunks": len(chunks),
        }

    def get_summary(self, resume_id) -> dict:
        """Return the stored summary for one resume_id with its identity.

        Mirrors ``get_full_resume``'s shape so the lookup tools can emit a
        minimal source record (resume_id + category) for the panel even
        though a one-line summary carries no chunk-level snippet or score.
        """
        records = self._scroll_by_resume_id(config.SUMMARY_COLLECTION, resume_id)
        if not records:
            raise KeyError(f"No summary found for resume_id {resume_id!r}.")
        payload = records[0].payload
        return {
            "resume_id": payload["metadata"]["resume_id"],
            "category": payload["metadata"]["category"],
            "summary": payload["page_content"],
        }
