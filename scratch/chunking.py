"""Step 1 — Chunking: split documents into passages small enough to embed and cite.

Strategy (markdown-aware, no LLM needed):
  1. Split the document into sections at headings, remembering the heading path
     ("Travel Policy > Hotels"), so every chunk knows where it came from.
  2. Pack whole paragraphs into chunks of at most `max_chars`.
  3. Carry the last paragraph into the next chunk of the same section (overlap),
     so a fact near a boundary is still retrievable with its context.
  4. Hard-split any single paragraph that is longer than `max_chars`.

Pure functions only: easy to test and to experiment with.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 300

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str  # "<document>#<n>", stable for a given document version
    document: str
    title: str  # heading path, e.g. "Travel and Expense Policy > Hotels"
    text: str


def chunk_markdown(
    document: str,
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> List[Chunk]:
    if max_chars <= overlap_chars:
        raise ValueError("max_chars must be larger than overlap_chars.")

    chunks: List[Chunk] = []
    for title, body in _sections(document, text):
        for passage in pack_paragraphs(split_paragraphs(body), max_chars, overlap_chars):
            chunks.append(Chunk(f"{document}#{len(chunks)}", document, title, passage))
    return chunks


def _sections(document: str, text: str) -> Iterator[Tuple[str, str]]:
    """Yield (heading path, body text) per section. Text before any heading uses the doc name."""
    path: List[Tuple[int, str]] = []  # (level, heading)
    body: List[str] = []

    def title() -> str:
        return " > ".join(h for _, h in path) or document

    for line in text.splitlines():
        match = _HEADING.match(line)
        if not match:
            body.append(line)
            continue
        if "\n".join(body).strip():
            yield title(), "\n".join(body)
        body = []
        level = len(match.group(1))
        path = [(lvl, h) for lvl, h in path if lvl < level] + [(level, match.group(2))]

    if "\n".join(body).strip():
        yield title(), "\n".join(body)


def split_paragraphs(body: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def pack_paragraphs(paragraphs: List[str], max_chars: int, overlap_chars: int) -> Iterator[str]:
    current: List[str] = []

    def size(parts: List[str]) -> int:
        return sum(len(p) for p in parts) + 2 * max(len(parts) - 1, 0)

    for paragraph in (piece for p in paragraphs for piece in _split_long(p, max_chars)):
        if current and size([*current, paragraph]) > max_chars:
            yield "\n\n".join(current)
            tail = current[-1]
            current = [tail] if len(tail) <= overlap_chars and size([tail, paragraph]) <= max_chars else []
        current.append(paragraph)

    if current:
        yield "\n\n".join(current)


def _split_long(paragraph: str, max_chars: int) -> Iterator[str]:
    """Split an oversized paragraph at sentence ends (or hard, as a last resort)."""
    if len(paragraph) <= max_chars:
        yield paragraph
        return
    piece = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:  # no sentence boundary to use
            if piece:
                yield piece
                piece = ""
            yield sentence[:max_chars]
            sentence = sentence[max_chars:]
        if piece and len(piece) + 1 + len(sentence) > max_chars:
            yield piece
            piece = sentence
        else:
            piece = f"{piece} {sentence}".strip()
    if piece:
        yield piece
