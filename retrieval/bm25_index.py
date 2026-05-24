"""
BM25 sparse retrieval index — Phase 2.

Builds over the `content` field of all indexed chunks (pulled from Milvus).
Persists to disk as a pickle so it survives process restarts without
needing to re-fetch from Milvus on every startup.

Must be explicitly rebuilt after new papers are ingested:
    index.build(vector_store.fetch_all_content())
    index.save()
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)


class BM25Index:
    """
    Wraps rank-bm25's BM25Okapi with chunk-id mapping and disk persistence.

    Typical usage after ingestion::

        index = BM25Index()
        index.build(vector_store.fetch_all_content())
        index.save()

    Typical usage at query time::

        index = BM25Index()
        if not index.load():
            index.build(vector_store.fetch_all_content())
            index.save()
        results = index.query("attention mechanism", top_k=10)
    """

    def __init__(self, index_path: Path = config.BM25_INDEX_PATH) -> None:
        self._index_path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: List[str] = []

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_built(self) -> bool:
        """True if the index is ready to query."""
        return self._bm25 is not None and bool(self._chunk_ids)

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunk_ids)

    # ------------------------------------------------------------------ #
    # Build / persist                                                      #
    # ------------------------------------------------------------------ #

    def build(self, items: List[Tuple[str, str]]) -> None:
        """
        (Re-)build the BM25 index from (chunk_id, content) pairs.

        Tokenises content with a simple whitespace split (lowercase).
        For English academic text this is a reasonable baseline; a more
        sophisticated tokeniser can be swapped in here later.

        Args:
            items: List of (chunk_id, content) tuples from
                   VectorStore.fetch_all_content().
        """
        if not items:
            logger.warning("BM25Index.build called with an empty items list.")
            return

        self._chunk_ids = [chunk_id for chunk_id, _ in items]
        tokenized = [content.lower().split() for _, content in items]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks.", len(items))

    def save(self, path: Path | None = None) -> None:
        """Persist index to disk (overwrites if already exists)."""
        dest = Path(path or self._index_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            pickle.dump({"chunk_ids": self._chunk_ids, "bm25": self._bm25}, fh)
        logger.info("BM25 index saved to %s  (%d chunks).", dest, len(self._chunk_ids))

    def load(self, path: Path | None = None) -> bool:
        """
        Load a previously persisted index.

        Returns:
            True if loaded successfully; False if no file was found.
        """
        src = Path(path or self._index_path)
        if not src.exists():
            logger.info("No BM25 index file found at %s — will build fresh.", src)
            return False

        with open(src, "rb") as fh:
            data = pickle.load(fh)

        self._chunk_ids = data["chunk_ids"]
        self._bm25 = data["bm25"]
        logger.info(
            "BM25 index loaded from %s  (%d chunks).", src, len(self._chunk_ids)
        )
        return True

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def query(self, text: str, top_k: int = config.TOP_K_BM25) -> List[Tuple[str, float]]:
        """
        Retrieve top-K chunks by BM25 score.

        Args:
            text:  Raw query string.  Tokenised identically to the corpus.
            top_k: Maximum number of results.

        Returns:
            List of (chunk_id, bm25_score) sorted by descending score.
            Returns [] if the index has not been built yet.
        """
        if not self.is_built:
            logger.warning("BM25Index.query called before index is built — returning [].")
            return []

        tokens = text.lower().split()
        scores: np.ndarray = self._bm25.get_scores(tokens)

        # argsort descending, clip to top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self._chunk_ids[int(i)], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0   # skip chunks with zero relevance
        ]
