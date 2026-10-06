"""Step 1b — Semantic chunking: let a small LLM decide where one topic ends and the next begins.

The markdown chunker (chunking.py) relies on headings and a size limit. That works for
tidy markdown, but text copied from web pages, PDFs, or transcripts often has no headings,
long paragraphs that switch topic halfway, and boilerplate (menus, footers, cookie notes),
so size-based cuts land mid-topic. Here a fast, cheap model (gemini-3.5-flash-lite by
default) reads the document and picks the boundaries instead.

How it works:
  1. Split the text into sentences (and lines, so headings and list items stand alone)
     and number them.
  2. The model returns only *where* each chunk starts, a descriptive title, and whether
     it is boilerplate, as JSON:
     {"chunks": [{"start": 1, "title": "...", "skip": true}, {"start": 3, "title": "...", "skip": false}]}
  3. We cut the ORIGINAL text at those sentences. The model never rewrites the content, so
     it can't paraphrase or invent facts, and the output stays small (fast and cheap).
     Boilerplate chunks are dropped.
  4. Boundaries are sanitized (range-checked, sorted, deduplicated) so every sentence lands
     in exactly one chunk whatever the model returns, and an oversized chunk is split
     further with the markdown chunker's paragraph packing.

The cost: one LLM call per document at index time (incremental sync skips unchanged
documents). Very long documents would need splitting into windows first.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from rag_common import chunking_model, get_client
from scratch.chunking import DEFAULT_MAX_CHARS, Chunk, pack_paragraphs, split_paragraphs
from translation import base_language, language_name

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Boundary:
    start: int  # number of the sentence where the chunk starts (1-based)
    title: str
    skip: bool = False  # boilerplate such as navigation or footers: not indexed


# (document name, numbered sentences "[12] text") -> boundaries
Segmenter = Callable[[str, str], List[Boundary]]

SEGMENT_INSTRUCTIONS = """\
You split a document into semantic chunks for a search index. The document is given as
numbered sentences. Return the chunks in order, each with the number of the sentence where it
starts, a title, and whether it is boilerplate.
- Each chunk covers one self-contained topic, so a question about that topic can be answered from it alone.
- Start a new chunk where the topic changes, even in the middle of a paragraph.
- Keep a heading with the text below it, a list with its intro sentence, and a table whole.
- Aim for roughly 200-1200 characters per chunk.
- Title: the document subject and the topic, e.g. "Travel policy - Hotel limits by city".
- skip=true only for content nobody would search for: site navigation, menus, footers,
  cookie or copyright notices. Everything else gets skip=false.
"""


def segment_instructions(base: str) -> str:
    """The chunking prompt, with titles in the base language whatever language the document is in.

    Titles are embedded with the chunk text, so one title language keeps the index consistent
    (see translation.py). Say it even for English: left alone, the model titles a Finnish
    document in Finnish."""
    name = language_name(base)
    return SEGMENT_INSTRUCTIONS + f"- Write every title in {name}, even when the document is in another language.\n"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "title": {"type": "string"},
                    "skip": {"type": "boolean"},
                },
                "required": ["start", "title", "skip"],
            },
        }
    },
    "required": ["chunks"],
}


class SemanticChunker:
    """Callable with the same signature as chunk_markdown: (document, text) -> chunks."""

    name = "semantic"

    def __init__(
        self,
        *,
        segment: Segmenter | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        base: str | None = None,
    ) -> None:
        self.base_language = base or base_language()
        self.segment = segment or gemini_segmenter(self.base_language)
        self.max_chars = max_chars

    def __call__(self, document: str, text: str) -> List[Chunk]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        numbered = "\n".join(f"[{n}] {sentence}" for n, (_, sentence) in enumerate(sentences, start=1))
        boundaries = self.segment(document, numbered)
        return chunks_from_boundaries(document, text, boundaries, max_chars=self.max_chars)


def split_sentences(text: str) -> List[Tuple[int, str]]:
    """[(character offset, sentence)] for every sentence, splitting at line breaks too."""
    sentences = []
    offset = 0
    for line in text.splitlines(keepends=True):
        position = 0
        for piece in _SENTENCE_END.split(line):
            position = line.index(piece, position)
            if piece.strip():
                sentences.append((offset + position, piece.strip()))
            position += len(piece)
        offset += len(line)
    return sentences


def gemini_segmenter(base: str) -> Segmenter:
    """A segmenter that asks the chunking model for chunk starts, titles (in the base language),
    and boilerplate flags as structured JSON."""

    def segment(document: str, numbered: str) -> List[Boundary]:
        from google.genai import types

        client = get_client()
        response = client.models.generate_content(
            model=chunking_model(),
            contents=f"Document: {document}\n\n{numbered}",
            config=types.GenerateContentConfig(
                system_instruction=segment_instructions(base),
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        try:
            chunks = json.loads(response.text or "")["chunks"]
            return [Boundary(int(c["start"]), str(c["title"]).strip(), bool(c.get("skip"))) for c in chunks]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Semantic chunking of {document} returned invalid JSON: {exc}") from exc

    return segment


def chunks_from_boundaries(
    document: str,
    text: str,
    boundaries: Sequence[Boundary],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Chunk]:
    """Cut the original text at the given boundaries (sentence numbers from split_sentences).

    Pure function. Tolerates whatever the model returns: out-of-range and duplicate starts
    are dropped and sentence 1 always starts a chunk, so every sentence ends up in exactly
    one chunk. Chunks marked skip are left out.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    starts: Dict[int, Boundary] = {}
    for boundary in sorted(boundaries, key=lambda b: b.start):
        if 1 <= boundary.start <= len(sentences):
            starts.setdefault(boundary.start, boundary)
    starts.setdefault(1, Boundary(1, document))

    ordered = [starts[n] for n in sorted(starts)]
    ends = [sentences[b.start - 1][0] for b in ordered[1:]] + [len(text)]
    chunks: List[Chunk] = []
    for boundary, end in zip(ordered, ends):
        if boundary.skip:
            continue
        body = text[sentences[boundary.start - 1][0] : end].strip()
        pieces = [body] if len(body) <= max_chars else pack_paragraphs(split_paragraphs(body), max_chars, 0)
        for piece in pieces:
            chunks.append(Chunk(f"{document}#{len(chunks)}", document, boundary.title or document, piece))
    return chunks
