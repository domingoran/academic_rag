"""
Metadata enrichment stage.

Phase 1: validates and cleans chunks (ensures IDs, strips empties).
Phase 2+: keyword extraction, cross-reference detection, author disambiguation, etc.
"""
from __future__ import annotations

import logging
import uuid
from typing import List

from core.schemas import Chunk

logger = logging.getLogger(__name__)


def build_metadata(chunks: List[Chunk]) -> List[Chunk]:
    """
    Validate and lightly enrich a list of Chunks.

    Current operations:
      • Assigns a fresh UUID to any chunk missing a chunk_id.
      • Drops chunks whose content is empty or whitespace-only.

    Args:
        chunks: Raw chunks from the chunker.

    Returns:
        Cleaned, enriched list of Chunk objects.
    """
    enriched: List[Chunk] = []

    for chunk in chunks:
        # Ensure every chunk has an id
        if not chunk.chunk_id:
            chunk.chunk_id = str(uuid.uuid4())

        # Drop chunks with no usable content
        if not chunk.content or not chunk.content.strip():
            logger.debug("Dropping empty chunk (paper=%s)", chunk.paper_id)
            continue

        enriched.append(chunk)

    dropped = len(chunks) - len(enriched)
    if dropped:
        logger.info("Dropped %d empty chunk(s) during metadata build", dropped)

    return enriched
