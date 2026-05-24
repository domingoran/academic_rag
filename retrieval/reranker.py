"""
LLM-based reranker — Phase 2.

Asks the Ollama model to order candidate chunks by relevance to the query.
A single generate() call is made with all candidates; the response is
parsed as a comma-separated list of 1-based chunk numbers.

Fallback strategy
-----------------
If the LLM response cannot be parsed into a valid permutation the reranker
returns the candidates in their original order so retrieval never silently
degrades.  Partial orderings are handled: valid indices are placed first,
then any missing ones are appended in their original relative order.
"""
from __future__ import annotations

import logging
import re
from typing import List

import config
from core.schemas import Chunk
from llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Maximum characters of each chunk shown to the reranker.
# Keeps the prompt short while still giving the LLM enough signal.
_PASSAGE_PREVIEW_CHARS = 400

_PROMPT_TEMPLATE = """\
You are a relevance judge for an academic paper retrieval system.

Given the query and numbered passages below, rank the passages from MOST to \
LEAST relevant to the query.

Return ONLY a comma-separated list of the passage numbers in relevance order \
(most relevant first).  Include every number exactly once.
Example for 5 passages: 3, 1, 5, 2, 4

Query: {query}

Passages:
{passages}

Ranking (numbers only, comma-separated):"""


class Reranker:
    """
    Reranks a candidate list of Chunks using an Ollama LLM.

    Usage::

        reranker = Reranker(ollama_client)
        top5 = reranker.rerank("What is attention?", chunks)[:5]
    """

    def __init__(self, ollama_client: OllamaClient) -> None:
        self._client = ollama_client

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Reorder *chunks* by relevance to *query*.

        Args:
            query:  The user's question.
            chunks: Candidate chunks (typically 10–20 from hybrid search).

        Returns:
            The same chunks in a new order (most relevant first).
            Falls back to the input order on any error.
        """
        if len(chunks) <= 1:
            return chunks

        passages = "\n\n".join(
            f"[{i + 1}] {chunk.content[:_PASSAGE_PREVIEW_CHARS]}"
            for i, chunk in enumerate(chunks)
        )
        prompt = _PROMPT_TEMPLATE.format(query=query, passages=passages)

        try:
            response = self._client.generate(prompt)
            order = self._parse_ranking(response, len(chunks))
            logger.info(
                "Reranker: %d candidates → parsed order %s", len(chunks), order[:5]
            )
            return [chunks[i] for i in order]
        except Exception as exc:
            logger.warning("Reranker failed (%s) — using original order.", exc)
            return chunks

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_ranking(response: str, n: int) -> List[int]:
        """
        Parse a comma-separated ranking into 0-based indices.

        Args:
            response: LLM output string, e.g. "3, 1, 5, 2, 4"
            n:        Expected number of passages.

        Returns:
            0-based index list.  Always has length == n.
        """
        raw_nums = re.findall(r'\d+', response)
        # Convert to 0-based, drop out-of-range values
        seen: set = set()
        result: List[int] = []
        for tok in raw_nums:
            idx = int(tok) - 1          # 1-based → 0-based
            if 0 <= idx < n and idx not in seen:
                result.append(idx)
                seen.add(idx)

        # Append any missing indices in original order
        for idx in range(n):
            if idx not in seen:
                result.append(idx)

        return result
