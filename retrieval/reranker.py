"""
Cross-encoder reranker using BAAI/bge-reranker-v2-m3 — Phase 3.

Scores each (query, passage) pair with a HuggingFace cross-encoder and returns
candidates sorted by descending relevance score.  No LLM call required.

Falls back to the original order on any error so retrieval never silently fails.
"""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import CrossEncoder

import config
from core.schemas import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranks candidate chunks using a cross-encoder model.

    Usage::

        reranker = Reranker()
        top5 = reranker.rerank("What is attention?", chunks)[:5]
    """

    def __init__(self) -> None:
        logger.info("Loading reranker model: %s", config.RERANKER_MODEL)
        self._model = CrossEncoder(config.RERANKER_MODEL)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Reorder *chunks* by relevance to *query*.

        Args:
            query:  The user's question.
            chunks: Candidate chunks (typically 10-20 from hybrid search).

        Returns:
            The same chunks sorted by cross-encoder score (most relevant first).
            Falls back to the input order on any error.
        """
        if len(chunks) <= 1:
            return chunks

        pairs = [(query, chunk.content) for chunk in chunks]

        try:
            scores = self._model.predict(pairs)
            ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
            logger.info(
                "Reranker: %d candidates scored; top score %.4f",
                len(chunks), ranked[0][0],
            )
            return [chunk for _, chunk in ranked]
        except Exception as exc:
            logger.warning("Reranker failed (%s) — using original order.", exc)
            return chunks
