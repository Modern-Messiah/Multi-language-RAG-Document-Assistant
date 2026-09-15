"""
Table-Aware Document Splitter for RAG Pipeline.

Detects and preserves table structures (Markdown pipe tables, HTML tables)
during chunking to prevent splitting critical rows and column headers
across chunk boundaries. For tables that exceed chunk_size, headers are
automatically repeated on every chunk.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Regex pattern to match HTML tables: <table>...</table> (case-insensitive, multiline)
HTML_TABLE_PATTERN = re.compile(r"(<table\b[^>]*>.*?</table>)", re.IGNORECASE | re.DOTALL)

# Regex pattern to identify a markdown table header separator row like |:---|---:| or |---|---|
MD_TABLE_SEPARATOR_PATTERN = re.compile(
    r"^\s*\|?\s*(?::?-+:?\s*\|)+\s*(?::?-+:?\s*)?\|?\s*$"
)


def is_markdown_table_row(line: str) -> bool:
    """Check if a line looks like a markdown table row with pipe characters."""
    stripped = line.strip()
    if not stripped:
        return False
    # Must contain at least one pipe, and either start/end with pipe or have multiple columns
    if "|" not in stripped:
        return False
    parts = stripped.split("|")
    # If starts and ends with |, parts will have at least 3 elements
    return len(parts) >= 3 or (stripped.startswith("|") or stripped.endswith("|"))


def is_markdown_separator_row(line: str) -> bool:
    """Check if a line is a markdown table separator row (| --- | --- |)."""
    return bool(MD_TABLE_SEPARATOR_PATTERN.match(line.strip()))


class TableBlock:
    """Represents an extracted table block."""

    def __init__(
        self,
        table_type: str,  # 'markdown' or 'html'
        raw_text: str,
        header_text: Optional[str] = None,
        rows: Optional[List[str]] = None,
    ):
        self.table_type = table_type
        self.raw_text = raw_text.strip()
        self.header_text = header_text
        self.rows = rows or []

    def __len__(self) -> int:
        return len(self.raw_text)


class ContentSegment:
    """A segment of text that is either normal prose or a structured table."""

    def __init__(self, content: str, is_table: bool = False, table_block: Optional[TableBlock] = None):
        self.content = content
        self.is_table = is_table
        self.table_block = table_block


def extract_segments(text: str) -> List[ContentSegment]:
    """
    Parse document text into alternating segments of prose and table blocks.
    Supports both HTML <table> blocks and Markdown pipe tables.
    """
    if not text or not text.strip():
        return []

    # 1. First extract HTML tables if any exist
    html_matches = list(HTML_TABLE_PATTERN.finditer(text))
    if html_matches:
        segments: List[ContentSegment] = []
        last_idx = 0
        for match in html_matches:
            start, end = match.span()
            if start > last_idx:
                prose = text[last_idx:start]
                # Process prose for any markdown tables
                sub_segments = _extract_markdown_segments(prose)
                segments.extend(sub_segments)

            table_html = match.group(1)
            tbl = TableBlock(table_type="html", raw_text=table_html)
            segments.append(ContentSegment(content=table_html, is_table=True, table_block=tbl))
            last_idx = end

        if last_idx < len(text):
            trailing_prose = text[last_idx:]
            segments.extend(_extract_markdown_segments(trailing_prose))
        return segments

    # 2. No HTML tables, extract markdown tables directly
    return _extract_markdown_segments(text)


def _extract_markdown_segments(text: str) -> List[ContentSegment]:
    """Parse text into segments identifying markdown tables."""
    lines = text.split("\n")
    segments: List[ContentSegment] = []
    current_prose_lines: List[str] = []

    idx = 0
    n = len(lines)

    while idx < n:
        line = lines[idx]

        # Check if this line and the next line could form a markdown table
        # A markdown table must have a header row followed by a separator row
        if idx + 1 < n and is_markdown_table_row(line) and is_markdown_separator_row(lines[idx + 1]):
            # Flush accumulated prose
            if current_prose_lines:
                prose_text = "\n".join(current_prose_lines)
                if prose_text.strip():
                    segments.append(ContentSegment(content=prose_text, is_table=False))
                current_prose_lines = []

            # Extract table lines
            table_lines = [line, lines[idx + 1]]
            header_str = f"{line}\n{lines[idx + 1]}"
            idx += 2
            row_lines: List[str] = []

            while idx < n and is_markdown_table_row(lines[idx]):
                table_lines.append(lines[idx])
                row_lines.append(lines[idx])
                idx += 1

            raw_tbl = "\n".join(table_lines)
            tbl = TableBlock(
                table_type="markdown",
                raw_text=raw_tbl,
                header_text=header_str,
                rows=row_lines,
            )
            segments.append(ContentSegment(content=raw_tbl, is_table=True, table_block=tbl))
        else:
            current_prose_lines.append(line)
            idx += 1

    if current_prose_lines:
        prose_text = "\n".join(current_prose_lines)
        if prose_text.strip():
            segments.append(ContentSegment(content=prose_text, is_table=False))

    return segments


class TableAwareSplitter:
    """
    Splits text documents into semantic chunks with special handling for tables.

    Rules:
    - Tables that fit within `chunk_size` are never split across chunks.
    - Large tables exceeding `chunk_size` are split row-by-row, with the header
      and separator rows automatically repeated on every split chunk.
    - Metadata flags `contains_table: True` and `is_table: True` are attached to
      generated chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        fallback_splitter: Optional[callable] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback_splitter = fallback_splitter

    def split_text(self, text: str) -> List[Tuple[str, dict]]:
        """
        Split raw text into chunks with metadata attributes.

        Returns:
            List of tuples: (chunk_text, chunk_meta_dict)
        """
        if not text or not text.strip():
            return []

        segments = extract_segments(text)
        if not segments:
            return []

        # Pure prose fast-path: preserve standard splitter behavior completely
        if not any(seg.is_table for seg in segments) and self.fallback_splitter:
            raw_chunks = self.fallback_splitter(text)
            return [
                (c, {"contains_table": False, "is_table": False})
                for c in raw_chunks
            ]

        chunks: List[Tuple[str, dict]] = []
        current_chunk_parts: List[str] = []
        current_chunk_len = 0
        current_chunk_has_table = False

        def flush_current():
            nonlocal current_chunk_parts, current_chunk_len, current_chunk_has_table
            if current_chunk_parts:
                joined = "\n\n".join(current_chunk_parts).strip()
                if joined:
                    chunks.append(
                        (
                            joined,
                            {
                                "contains_table": current_chunk_has_table,
                                "is_table": current_chunk_has_table and len(current_chunk_parts) == 1,
                            },
                        )
                    )
                current_chunk_parts = []
                current_chunk_len = 0
                current_chunk_has_table = False

        for seg in segments:
            if not seg.is_table:
                # Prose segment: split if too large, or accumulate
                prose_subchunks = self._split_prose(seg.content)
                for p_chunk in prose_subchunks:
                    p_len = len(p_chunk)
                    if current_chunk_len + p_len + 2 <= self.chunk_size:
                        current_chunk_parts.append(p_chunk)
                        current_chunk_len += p_len + 2
                    else:
                        flush_current()
                        current_chunk_parts.append(p_chunk)
                        current_chunk_len = p_len
            else:
                # Table segment
                tbl = seg.table_block
                tbl_text = seg.content
                tbl_len = len(tbl_text)

                if tbl_len <= self.chunk_size:
                    # Fits in a single chunk
                    if current_chunk_len + tbl_len + 2 <= self.chunk_size:
                        current_chunk_parts.append(tbl_text)
                        current_chunk_len += tbl_len + 2
                        current_chunk_has_table = True
                    else:
                        flush_current()
                        current_chunk_parts.append(tbl_text)
                        current_chunk_len = tbl_len
                        current_chunk_has_table = True
                else:
                    # Table is larger than chunk_size: flush current chunk and split table with header
                    flush_current()
                    table_chunks = self._split_large_table(tbl)
                    for t_chunk in table_chunks:
                        chunks.append(
                            (
                                t_chunk,
                                {
                                    "contains_table": True,
                                    "is_table": True,
                                },
                            )
                        )

        flush_current()
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using table-aware chunking while preserving metadata.
        """
        if not documents:
            return []

        all_chunks: List[Document] = []
        for doc in documents:
            text = doc.page_content or ""
            chunks_with_meta = self.split_text(text)

            for chunk_text, table_meta in chunks_with_meta:
                meta = dict(doc.metadata)
                meta.update(table_meta)
                all_chunks.append(Document(page_content=chunk_text, metadata=meta))

        # Assign upload-wide sequential chunk_id and total_chunks across the entire batch
        total = len(all_chunks)
        for idx, chunk in enumerate(all_chunks):
            chunk.metadata.update(
                {
                    "chunk_id": idx,
                    "chunk_size": len(chunk.page_content),
                    "total_chunks": total,
                }
            )

        return all_chunks

    def _split_prose(self, text: str) -> List[str]:
        """Split prose text using fallback splitter or paragraph breaks."""
        if self.fallback_splitter:
            return self.fallback_splitter(text)

        # Standard paragraph/sentence splitting
        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            if cur_len + len(p_strip) + 2 <= self.chunk_size:
                cur.append(p_strip)
                cur_len += len(p_strip) + 2
            else:
                if cur:
                    chunks.append("\n\n".join(cur))
                if len(p_strip) > self.chunk_size:
                    # Break huge paragraph by single newlines or words
                    lines = p_strip.split("\n")
                    sub_cur: List[str] = []
                    sub_len = 0
                    for line in lines:
                        if sub_len + len(line) + 1 <= self.chunk_size:
                            sub_cur.append(line)
                            sub_len += len(line) + 1
                        else:
                            if sub_cur:
                                chunks.append("\n".join(sub_cur))
                            sub_cur = [line]
                            sub_len = len(line)
                    if sub_cur:
                        cur = sub_cur
                        cur_len = sub_len
                    else:
                        cur = []
                        cur_len = 0
                else:
                    cur = [p_strip]
                    cur_len = len(p_strip)

        if cur:
            chunks.append("\n\n".join(cur))

        return chunks

    def _split_large_table(self, table_block: Optional[TableBlock]) -> List[str]:
        """
        Split a table exceeding chunk_size into smaller chunks, repeating
        the table header on each slice to preserve semantic context.
        """
        if not table_block:
            return []

        # If it's HTML, we can slice <tr> elements or return clean slices
        if table_block.table_type == "html":
            return self._split_large_html_table(table_block.raw_text)

        header = table_block.header_text or ""
        rows = table_block.rows
        if not rows:
            return [table_block.raw_text]

        header_len = len(header) + 1  # newline
        available_size = max(self.chunk_size - header_len, 100)

        chunks: List[str] = []
        current_rows: List[str] = []
        current_size = 0

        for row in rows:
            row_len = len(row) + 1
            if current_size + row_len <= available_size:
                current_rows.append(row)
                current_size += row_len
            else:
                if current_rows:
                    chunk_text = f"{header}\n" + "\n".join(current_rows)
                    chunks.append(chunk_text)
                current_rows = [row]
                current_size = row_len

        if current_rows:
            chunk_text = f"{header}\n" + "\n".join(current_rows)
            chunks.append(chunk_text)

        return chunks

    def _split_large_html_table(self, html_text: str) -> List[str]:
        """Split a large HTML table across <tr> rows."""
        # Find <thead> or first <tr> as header
        thead_match = re.search(r"(<thead\b[^>]*>.*?</thead>)", html_text, re.IGNORECASE | re.DOTALL)
        header_html = thead_match.group(1) if thead_match else ""

        # Extract all <tr> rows
        row_matches = re.findall(r"(<tr\b[^>]*>.*?</tr>)", html_text, re.IGNORECASE | re.DOTALL)
        if not row_matches:
            # Fallback simple string slice
            return [html_text[i : i + self.chunk_size] for i in range(0, len(html_text), self.chunk_size)]

        if not header_html and row_matches:
            header_html = row_matches[0]
            row_matches = row_matches[1:]

        chunks: List[str] = []
        current_rows: List[str] = []
        current_len = len(header_html) + 30  # <table>...</table> overhead

        for row in row_matches:
            row_len = len(row)
            if current_len + row_len <= self.chunk_size:
                current_rows.append(row)
                current_len += row_len
            else:
                if current_rows:
                    chunk_html = f"<table>\n{header_html}\n<tbody>\n" + "\n".join(current_rows) + "\n</tbody>\n</table>"
                    chunks.append(chunk_html)
                current_rows = [row]
                current_len = len(header_html) + 30 + row_len

        if current_rows:
            chunk_html = f"<table>\n{header_html}\n<tbody>\n" + "\n".join(current_rows) + "\n</tbody>\n</table>"
            chunks.append(chunk_html)

        return chunks
