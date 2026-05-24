"""
Structure-aware + token-budget chunking.

Rules (from CLAUDE.md §6):
  • Section headers are primary split points.
  • Max token budget per chunk: CHUNK_MAX_TOKENS (≈ 512 tokens).
  • Tables  → one chunk each.
  • Figures → one chunk (caption only); skipped if no caption.
  • Equations / formulas → standalone chunk.
  • Text within a section is accumulated; split with overlap if over budget.
"""
from __future__ import annotations

import logging
from typing import List

import config
from core.schemas import Chunk, ChunkMetadata, ParsedDocument, RawElement
from docling_core.types.doc import DocItemLabel

logger = logging.getLogger(__name__)

# Labels that mark the start of a new section
_SECTION_LABELS = {
    DocItemLabel.SECTION_HEADER.value,
}

# Labels whose text should accumulate into a running text buffer
_TEXT_LABELS = {
    DocItemLabel.TEXT.value,
    DocItemLabel.LIST_ITEM.value,
    DocItemLabel.PARAGRAPH.value,
    DocItemLabel.FOOTNOTE.value,
    DocItemLabel.CODE.value,
    # Note: CAPTION is handled inside table/figure branches
}

# Max chars stored in a single Milvus VARCHAR(8192) content field
_MAX_CONTENT_CHARS = 8000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 0.75 words for English."""
    return int(len(text.split()) / 0.75)


def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split *text* into word-level chunks that stay under *max_tokens*.
    Adjacent chunks share *overlap_tokens* words for context continuity.
    """
    words = text.split()
    # Convert token budgets to approximate word counts
    words_per_chunk = max(1, int(max_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))

    if len(words) <= words_per_chunk:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(words):
            break
        start = end - overlap_words
        if start < 0:
            start = 0

    return chunks


def chunk_document(parsed_doc: ParsedDocument) -> List[Chunk]:
    """
    Convert a ParsedDocument into a flat list of Chunks.

    Args:
        parsed_doc: output of the docling parser.

    Returns:
        List of Chunk objects (without embeddings).
    """
    chunks: List[Chunk] = []
    current_section = "Abstract"   # reasonable default before first header
    text_buffer: List[str] = []
    last_page: int = 0

    def _make_chunk(
        chunk_type: str,
        content: str,
        section: str,
        page: int,
    ) -> Chunk:
        return Chunk(
            paper_id=parsed_doc.paper_id,
            title=parsed_doc.title,
            authors=parsed_doc.authors,
            year=parsed_doc.year,
            section=section,
            chunk_type=chunk_type,
            content=content[:_MAX_CONTENT_CHARS],
            metadata=ChunkMetadata(page=page),
        )

    def _flush_text_buffer(section: str, page: int) -> None:
        nonlocal text_buffer
        if not text_buffer:
            return
        full_text = " ".join(text_buffer).strip()
        text_buffer = []
        if not full_text:
            return

        for sub in _split_text(full_text, config.CHUNK_MAX_TOKENS, config.CHUNK_OVERLAP_TOKENS):
            if sub:
                chunks.append(_make_chunk("text", sub, section, page))

    for elem in parsed_doc.elements:
        page = elem.page or last_page
        last_page = page

        # ---------------------------------------------------------------- #
        # Section header → flush current buffer, update running section   #
        # ---------------------------------------------------------------- #
        if elem.label in _SECTION_LABELS:
            _flush_text_buffer(current_section, page)
            if elem.text:
                current_section = elem.text.strip()
            continue

        # ---------------------------------------------------------------- #
        # Title (document title, not a section header)                    #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TITLE.value:
            # Already captured as parsed_doc.title; skip to avoid duplication
            continue

        # ---------------------------------------------------------------- #
        # Plain text → accumulate                                          #
        # ---------------------------------------------------------------- #
        if elem.label in _TEXT_LABELS:
            if elem.text:
                text_buffer.append(elem.text)
            continue

        # ---------------------------------------------------------------- #
        # Table → flush + standalone chunk                                 #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TABLE.value:
            _flush_text_buffer(current_section, page)
            content_parts: List[str] = []
            if elem.caption:
                content_parts.append(f"Table: {elem.caption}")
            if elem.markdown:
                content_parts.append(elem.markdown)
            content = "\n\n".join(content_parts).strip()
            if content:
                chunks.append(_make_chunk("table", content, current_section, page))
            continue

        # ---------------------------------------------------------------- #
        # Figure / Picture → caption-only chunk (skip if no caption)      #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.PICTURE.value:
            _flush_text_buffer(current_section, page)
            if elem.caption:
                chunks.append(_make_chunk(
                    "figure",
                    f"Figure: {elem.caption}",
                    current_section,
                    page,
                ))
            continue

        # ---------------------------------------------------------------- #
        # Formula / Equation → standalone chunk                           #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.FORMULA.value:
            if elem.text:
                chunks.append(_make_chunk("equation", elem.text, current_section, page))
            continue

        # ---------------------------------------------------------------- #
        # Anything else with text → treat as generic text                 #
        # ---------------------------------------------------------------- #
        if elem.text:
            text_buffer.append(elem.text)

    # Flush any remaining accumulated text
    _flush_text_buffer(current_section, last_page)

    logger.info(
        "Chunked '%s' → %d chunks (paper_id=%s)",
        parsed_doc.title,
        len(chunks),
        parsed_doc.paper_id,
    )
    return chunks
