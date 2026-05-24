"""
Structure-aware + token-budget chunking.

Rules:
  • Section headers are primary split points.
  • Max token budget per chunk: CHUNK_MAX_TOKENS (≈ 512 tokens).
  • Tables     → one chunk each; surrounding prose stitched in as context.
  • Figures    → one chunk (with or without caption); context stitched in.
  • Equations  → standalone chunk; surrounding prose stitched in as context.
  • Text within a section is accumulated; split with overlap if over budget.

Phase 3 improvements
--------------------
  1. Context stitching: tables, figures, and equations include _CONTEXT_WORDS
     words of immediately preceding and following prose.  This gives the
     embedding model the semantic signal that normally lives around the element
     rather than just the raw content (which is often too sparse to embed well).
  2. Figures without captions are no longer silently dropped; a chunk is still
     created when surrounding context is available.
  3. Sequential IDs (tbl-N, fig-N, eq-N) are stored in ChunkMetadata so
     individual elements can be targeted by metadata filters.
  4. Large table markdown is truncated intelligently: the header row is kept
     and a row-count note is appended rather than slicing mid-cell.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import config
from core.schemas import Chunk, ChunkMetadata, ParsedDocument, RawElement
from docling_core.types.doc import DocItemLabel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------

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
    # CAPTION is handled inside the table / figure branches
}

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Milvus VARCHAR(8192) limit
_MAX_CONTENT_CHARS = 8_000

# Words of surrounding prose to stitch into table / figure / equation chunks
_CONTEXT_WORDS = 50

# Table markdown is capped at this before context is added so the header rows
# always survive and context snippets are never silently cut off by the
# _MAX_CONTENT_CHARS limit.
_MAX_TABLE_MARKDOWN_CHARS = 6_000

# Regex that matches the separator row of a Markdown table: |---|---|
_TABLE_SEP_RE = re.compile(r'\|[-| :]+\|\s*\n')


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """
    Estimate the token count for an English text using a words-to-tokens heuristic.
    
    Returns:
        int: Estimated number of tokens in `text`, computed as len(text.split()) / 0.75 and truncated to an integer.
    """
    return int(len(text.split()) / 0.75)


def _split_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split text into word-based chunks with shared overlap to respect a token budget.
    
    Parameters:
        text (str): Input text to split.
        max_tokens (int): Target maximum token budget per chunk (word-based splitting).
        overlap_tokens (int): Number of tokens worth of overlap to include between adjacent chunks.
    
    Returns:
        List[str]: A list of non-empty chunk strings. Returns an empty list if the input is blank or contains only whitespace.
    """
    words = text.split()
    words_per_chunk = max(1, int(max_tokens * 0.75))
    overlap_words   = max(0, int(overlap_tokens * 0.75))

    if len(words) <= words_per_chunk:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end        = min(start + words_per_chunk, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end == len(words):
            break
        start = end - overlap_words
        if start < 0:
            start = 0

    return chunks


def _preceding_context(text_buffer: List[str]) -> str:
    """
    Get the last _CONTEXT_WORDS words from the accumulated text buffer as a single space-separated string.
    
    Returns:
        str: The last `_CONTEXT_WORDS` words from `text_buffer` joined by spaces, or an empty string if the buffer contains no words.
    """
    all_text = " ".join(text_buffer)
    words    = all_text.split()
    if not words:
        return ""
    return " ".join(words[-_CONTEXT_WORDS:])


def _following_context(elements: List[RawElement], start_idx: int) -> str:
    """
    Collect up to _CONTEXT_WORDS words of prose from elements that follow start_idx, stopping early at the next section header.
    
    Parameters:
        elements (List[RawElement]): Sequence of document elements to scan.
        start_idx (int): Index to start looking after; scanning begins at start_idx + 1.
    
    Returns:
        str: A single string of up to _CONTEXT_WORDS words drawn from subsequent TEXT-like elements (scanned across at most 20 following elements), stopping if a section header is encountered.
    """
    collected: List[str] = []
    for j in range(start_idx + 1, min(start_idx + 20, len(elements))):
        el = elements[j]
        if el.label in _SECTION_LABELS:
            break
        if el.label in _TEXT_LABELS and el.text:
            collected.extend(el.text.split())
            if len(collected) >= _CONTEXT_WORDS:
                break
    return " ".join(collected[:_CONTEXT_WORDS])


def _with_context(core: str, preceding: str, following: str) -> str:
    """
    Wrap a core text block with optional preceding and following prose snippets for context.
    
    Parameters:
        core (str): Main content to include unmodified.
        preceding (str): Prose to place before `core`; if non-empty it is wrapped in `[...]`.
        following (str): Prose to place after `core`; if non-empty it is wrapped in `[...]`.
    
    Returns:
        str: The assembled content where optional preceding and following snippets are each surrounded
        by `[...]` and separated from the core by blank lines. If a snippet is empty it is omitted.
    """
    parts: List[str] = []
    if preceding:
        parts.append(f"[...{preceding}...]")
    parts.append(core)
    if following:
        parts.append(f"[...{following}...]")
    return "\n\n".join(parts)


def _truncate_table_markdown(markdown: str) -> str:
    """
    Truncates a Markdown table to fit within the module's character budget while preserving the header and as many data rows as possible.
    
    Parameters:
        markdown (str): The table in Markdown format.
    
    Returns:
        str: The original Markdown if it fits within the character limit; otherwise a truncated Markdown string. If a Markdown separator row is found, the header (through the separator) is preserved and as many following non-empty rows are kept as fit within the budget, with a trailing "[... N rows truncated]" note when rows are dropped. If no header/separator is found, the function returns a character-truncated prefix with a trailing "[... truncated]" note.
    """
    if len(markdown) <= _MAX_TABLE_MARKDOWN_CHARS:
        return markdown

    match = _TABLE_SEP_RE.search(markdown)
    if match:
        header_end = match.end()
        header     = markdown[:header_end]
        rows       = [r for r in markdown[header_end:].splitlines() if r.strip()]

        budget   = _MAX_TABLE_MARKDOWN_CHARS - len(header) - 40   # room for note
        included: List[str] = []
        used = 0
        for row in rows:
            row_len = len(row) + 1   # +1 for the newline
            if used + row_len > budget:
                break
            included.append(row)
            used += row_len

        dropped = len(rows) - len(included)
        result  = header + "\n".join(included)
        if dropped > 0:
            result += f"\n[... {dropped} rows truncated]"
        return result

    # No recognisable Markdown header — plain truncate
    return markdown[:_MAX_TABLE_MARKDOWN_CHARS] + "\n[... truncated]"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chunk_document(parsed_doc: ParsedDocument) -> List[Chunk]:
    """
    Split a parsed document into structure-aware chunks suitable for embedding and downstream indexing.
    
    Converts the provided ParsedDocument into a flat list of Chunk objects by:
    - Accumulating prose into section-aware text chunks and splitting them to respect the token budget with overlap.
    - Emitting standalone, context-stitched chunks for tables, figures, and equations (including surrounding preceding/following prose where available).
    - Truncating large table markdown intelligently and limiting final chunk content size.
    - Attaching per-document sequential IDs for tables (tbl-N), figures (fig-N), and equations (eq-N) in chunk metadata.
    
    Parameters:
        parsed_doc (ParsedDocument): Parsed document containing elements, paper metadata, and element-level fields used to build chunks.
    
    Returns:
        List[Chunk]: Flat list of Chunk objects (content truncated to module limits; chunks do not include embeddings).
    """
    chunks: List[Chunk] = []
    current_section = "Abstract"   # reasonable default before first header
    text_buffer: List[str] = []
    last_page: int = 0

    # Per-document sequential counters for element IDs
    table_counter    = 0
    figure_counter   = 0
    equation_counter = 0

    elements   = parsed_doc.elements
    n_elements = len(elements)

    # ---------------------------------------------------------------------- #
    # Inner helpers (closures over parsed_doc / chunks / text_buffer)        #
    # ---------------------------------------------------------------------- #

    def _make_chunk(
        chunk_type: str,
        content: str,
        section: str,
        page: int,
        meta: Optional[ChunkMetadata] = None,
    ) -> Chunk:
        """
        Create a Chunk populated with the enclosing document's metadata and the supplied content.
        
        Parameters:
            chunk_type (str): Logical type for the chunk (e.g., "text", "table", "figure", "equation").
            content (str): Chunk body; will be truncated to _MAX_CONTENT_CHARS characters.
            section (str): Section name to associate with the chunk.
            page (int): Page number to store in chunk metadata when no explicit metadata is provided.
            meta (Optional[ChunkMetadata]): Optional metadata override; if omitted, a ChunkMetadata with the provided `page` is used.
        
        Returns:
            Chunk: A Chunk whose paper-level fields are copied from the enclosing parsed document, with the given section, chunk_type, truncated content, and metadata.
        """
        return Chunk(
            paper_id   = parsed_doc.paper_id,
            title      = parsed_doc.title,
            authors    = parsed_doc.authors,
            year       = parsed_doc.year,
            section    = section,
            chunk_type = chunk_type,
            content    = content[:_MAX_CONTENT_CHARS],
            metadata   = meta if meta is not None else ChunkMetadata(page=page),
        )

    def _flush_text_buffer(section: str, page: int) -> None:
        """
        Flushes the accumulated prose buffer into one or more text chunks for the given section and page.
        
        Joins and clears the internal text buffer, splits the resulting string into token-bounded subchunks using the module's chunking configuration, and appends each non-empty subchunk as a "text" Chunk with the provided section and page metadata. This function mutates the shared text_buffer and the chunks list.
        
        Parameters:
            section (str): Section name to assign to created chunks.
            page (int): Page number to record in each chunk's metadata.
        """
        nonlocal text_buffer
        if not text_buffer:
            return
        full_text  = " ".join(text_buffer).strip()
        text_buffer = []
        if not full_text:
            return
        for sub in _split_text(full_text, config.CHUNK_MAX_TOKENS, config.CHUNK_OVERLAP_TOKENS):
            if sub:
                chunks.append(_make_chunk("text", sub, section, page))

    # ---------------------------------------------------------------------- #
    # Main loop                                                               #
    # ---------------------------------------------------------------------- #

    for i, elem in enumerate(elements):
        page      = elem.page or last_page
        last_page = page

        # ---------------------------------------------------------------- #
        # Section header → flush buffer, update running section            #
        # ---------------------------------------------------------------- #
        if elem.label in _SECTION_LABELS:
            _flush_text_buffer(current_section, page)
            if elem.text:
                current_section = elem.text.strip()
            continue

        # ---------------------------------------------------------------- #
        # Title — already captured as parsed_doc.title; skip              #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TITLE.value:
            continue

        # ---------------------------------------------------------------- #
        # Plain text → accumulate                                          #
        # ---------------------------------------------------------------- #
        if elem.label in _TEXT_LABELS:
            if elem.text:
                text_buffer.append(elem.text)
            continue

        # ---------------------------------------------------------------- #
        # Table                                                             #
        # → flush buffer                                                   #
        # → context-stitched standalone chunk with smart truncation        #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.TABLE.value:
            table_counter += 1
            table_id = f"tbl-{table_counter}"

            pre = _preceding_context(text_buffer)
            _flush_text_buffer(current_section, page)
            fol = _following_context(elements, i)

            core_parts: List[str] = []
            if elem.caption:
                core_parts.append(f"Table: {elem.caption}")
            if elem.markdown:
                core_parts.append(_truncate_table_markdown(elem.markdown))
            core = "\n\n".join(core_parts).strip()

            if core:
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, table_id=table_id)
                chunks.append(_make_chunk("table", content, current_section, page, meta))
            continue

        # ---------------------------------------------------------------- #
        # Figure / Picture                                                  #
        # → flush buffer                                                   #
        # → context-stitched chunk; no-caption figures kept if context     #
        #   is available                                                    #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.PICTURE.value:
            figure_counter += 1
            figure_id = f"fig-{figure_counter}"

            pre = _preceding_context(text_buffer)
            _flush_text_buffer(current_section, page)
            fol = _following_context(elements, i)

            core = (
                f"Figure: {elem.caption}"
                if elem.caption
                else "Figure (no caption)"
            )

            # Only discard when there is truly nothing useful to embed
            if elem.caption or pre or fol:
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, figure_id=figure_id)
                chunks.append(_make_chunk("figure", content, current_section, page, meta))
            continue

        # ---------------------------------------------------------------- #
        # Formula / Equation                                                #
        # → context-stitched standalone chunk                              #
        # NOTE: buffer is NOT flushed — equations are inline elements and  #
        #       the surrounding text flow must continue uninterrupted.     #
        # ---------------------------------------------------------------- #
        if elem.label == DocItemLabel.FORMULA.value:
            equation_counter += 1
            equation_id = f"eq-{equation_counter}"

            pre = _preceding_context(text_buffer)
            fol = _following_context(elements, i)

            core = f"Equation: {elem.text}" if elem.text else "Equation"

            if elem.text or pre or fol:
                content = _with_context(core, pre, fol)
                meta    = ChunkMetadata(page=page, equation_id=equation_id)
                chunks.append(_make_chunk("equation", content, current_section, page, meta))
            continue

        # ---------------------------------------------------------------- #
        # Anything else with text → treat as generic text                 #
        # ---------------------------------------------------------------- #
        if elem.text:
            text_buffer.append(elem.text)

    # Flush any remaining accumulated text
    _flush_text_buffer(current_section, last_page)

    logger.info(
        "Chunked '%s' → %d chunks  "
        "(tables=%d, figures=%d, equations=%d, paper_id=%s)",
        parsed_doc.title,
        len(chunks),
        table_counter,
        figure_counter,
        equation_counter,
        parsed_doc.paper_id,
    )
    return chunks
