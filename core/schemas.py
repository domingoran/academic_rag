"""
Pydantic data models shared across the pipeline.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document-level models (output of the ingestion / parser stage)
# ---------------------------------------------------------------------------

class RawElement(BaseModel):
    """
    One structural element extracted from a PDF by Docling.
    The chunker reads a list of these to produce Chunks.
    """
    label: str            # DocItemLabel value: 'text', 'section_header', 'table', …
    text: Optional[str] = None        # plain text content
    markdown: Optional[str] = None    # markdown rendering (tables)
    caption: Optional[str] = None     # figure / table caption
    page: int = 0                     # 1-based page number (0 = unknown)
    level: int = 0                    # nesting depth from iterate_items


class ParsedDocument(BaseModel):
    """
    Document-level metadata + ordered list of raw structural elements.
    Produced by the parser; consumed by the chunker.
    """
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: int = 0
    elements: List[RawElement] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunk-level models (output of the chunker + embedder stages)
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Per-chunk provenance metadata stored in Milvus scalar fields."""
    page: int = 0
    figure_id: Optional[str] = None
    table_id: Optional[str] = None
    equation_id: Optional[str] = None


class Chunk(BaseModel):
    """
    The core unit of retrieval.  One Chunk = one Milvus entity.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: int = 0
    section: str = ""
    chunk_type: str = "text"   # text | table | figure | equation
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    # Populated by the embedder; not stored as a Pydantic field in Milvus
    embedding: Optional[List[float]] = None
