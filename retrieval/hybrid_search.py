"""
Hybrid retrieval: weighted fusion of dense vector search and BM25 — Phase 2.

Score fusion pipeline
---------------------
1. Fetch up to TOP_K_VECTOR results from Milvus (cosine scores in [0, 1]).
2. Fetch up to TOP_K_BM25  results from BM25   (unbounded positive floats).
3. Each score list is min-max normalised independently to [0, 1].
   Chunks that appear in only one list receive 0 in the missing component.
4. Final score = HYBRID_VECTOR_WEIGHT * vec_norm + HYBRID_BM25_WEIGHT * bm25_norm
5. Return top-K by fused score.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import config
from core.schemas import Chunk
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _minmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a {id: score} dict to [0, 1]. Handles edge cases."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        # All scores identical — map everything to 1.0
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


class HybridSearcher:
    """
    Combines dense vector search with BM25 sparse retrieval via
    weighted score fusion.

    Usage::

        searcher = HybridSearcher(vector_store, bm25_index)
        chunks = searcher.search(query_embedding, query_text, top_k=5)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        vector_weight: float = config.HYBRID_VECTOR_WEIGHT,
        bm25_weight: float = config.HYBRID_BM25_WEIGHT,
    ) -> None:
        self._vs = vector_store
        self._bm25 = bm25_index
        self._vec_w = vector_weight
        self._bm25_w = bm25_weight

    def search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = config.TOP_K_RERANK,
        expr: Optional[str] = None,
    ) -> List[Chunk]:
        """
        Retrieve and fuse results from vector search and BM25.

        If the BM25 index is not built (e.g. first run), falls back to
        pure vector search transparently.

        Args:
            query_embedding: Dense query vector.
            query_text:      Raw query string for BM25 tokenisation.
            top_k:           Number of final results to return.
            expr:            Optional Milvus scalar filter expression, e.g.
                             'year >= 2022 && chunk_type == "text"'

        Returns:
            Fused and ranked list of Chunk objects (length ≤ top_k).
        """
        # ----------------------------------------------------------------
        # 1. Dense retrieval
        # ----------------------------------------------------------------
        vec_pairs: List[Tuple[Chunk, float]] = self._vs.search_with_scores(
            query_embedding,
            top_k=config.TOP_K_VECTOR,
            expr=expr,
        )

        vec_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        for chunk, score in vec_pairs:
            vec_scores[chunk.chunk_id] = score
            chunk_map[chunk.chunk_id] = chunk

        # ----------------------------------------------------------------
        # 2. Sparse (BM25) retrieval
        # ----------------------------------------------------------------
        bm25_scores: Dict[str, float] = {}

        if self._bm25.is_built:
            bm25_hits = self._bm25.query(query_text, top_k=config.TOP_K_BM25)
            for cid, score in bm25_hits:
                bm25_scores[cid] = score

            # Materialise BM25 hits that weren't returned by vector search
            missing_ids = [cid for cid in bm25_scores if cid not in chunk_map]
            if missing_ids:
                for chunk in self._vs.fetch_by_ids(missing_ids):
                    chunk_map[chunk.chunk_id] = chunk
        else:
            logger.warning(
                "BM25 index not built — falling back to pure vector search."
            )

        # ----------------------------------------------------------------
        # 3. Score fusion
        # ----------------------------------------------------------------
        vec_norm  = _minmax(vec_scores)
        bm25_norm = _minmax(bm25_scores)

        all_ids = set(vec_norm) | set(bm25_norm)
        fused: Dict[str, float] = {
            cid: (
                self._vec_w  * vec_norm.get(cid, 0.0)
                + self._bm25_w * bm25_norm.get(cid, 0.0)
            )
            for cid in all_ids
        }

        # ----------------------------------------------------------------
        # 4. Sort, clip, return
        # ----------------------------------------------------------------
        sorted_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]

        results = [chunk_map[cid] for cid in sorted_ids if cid in chunk_map]
        logger.info(
            "HybridSearcher: vec=%d  bm25=%d  fused=%d  returned=%d",
            len(vec_scores),
            len(bm25_scores),
            len(fused),
            len(results),
        )
        return results
