"""
Embedding generation using HuggingFace sentence-transformers.

Default model: BAAI/bge-base-en-v1.5 (768-dim, configurable in config.py).

BGE retrieval convention:
  • Document passages  → encode as-is.
  • Queries            → prepend EMBEDDING_QUERY_INSTRUCTION before encoding.
  Set EMBEDDING_QUERY_INSTRUCTION = "" in config.py to disable for other models.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class Embedder:
    """
    Thin wrapper around a SentenceTransformer model.

    Usage::

        embedder = Embedder()
        doc_embeddings = embedder.embed_passages(["text one", "text two"])
        query_embedding = embedder.embed_query("what is attention?")
    """

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        query_instruction: str = config.EMBEDDING_QUERY_INSTRUCTION,
    ) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document passages (no instruction prefix).

        Args:
            texts: Non-empty list of passage strings.

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []

        logger.info("Embedding %d passage(s) in batches of %d", len(texts), self.batch_size)
        all_embeddings: List[List[float]] = []

        for start in tqdm(range(0, len(texts), self.batch_size), desc="Embedding", unit="batch"):
            batch = texts[start : start + self.batch_size]
            vecs = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.extend(vecs.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single retrieval query.
        If EMBEDDING_QUERY_INSTRUCTION is non-empty, it is prepended to the query.

        Args:
            query: User's natural-language question.

        Returns:
            Float vector of length EMBEDDING_DIM.
        """
        text = (self.query_instruction + query) if self.query_instruction else query
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
