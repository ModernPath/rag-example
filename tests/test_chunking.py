from __future__ import annotations

import pytest

from rag_common import DEFAULT_DOCS_DIR, load_documents
from scratch.chunking import chunk_markdown

DOC = """\
# Travel Policy

Intro text before any subsection.

## Hotels

London: 230 EUR.

### Exceptions

Ask your manager.

## Flights

Economy by default.
"""


def test_heading_paths_and_ids():
    chunks = chunk_markdown("travel.md", DOC)
    assert [c.title for c in chunks] == [
        "Travel Policy",
        "Travel Policy > Hotels",
        "Travel Policy > Hotels > Exceptions",
        "Travel Policy > Flights",  # deeper heading popped when a sibling appears
    ]
    assert [c.id for c in chunks] == [f"travel.md#{i}" for i in range(4)]
    assert chunks[1].text == "London: 230 EUR."


def test_text_without_headings_uses_document_name():
    chunks = chunk_markdown("notes.txt", "Just one paragraph.")
    assert [(c.title, c.text) for c in chunks] == [("notes.txt", "Just one paragraph.")]


def test_packs_paragraphs_up_to_max_chars_with_overlap():
    paragraphs = [f"Paragraph {i} " + "x" * 80 for i in range(6)]  # ~92 chars each
    chunks = chunk_markdown("d.md", "# T\n\n" + "\n\n".join(paragraphs), max_chars=300, overlap_chars=100)
    assert all(len(c.text) <= 300 for c in chunks)
    assert len(chunks) > 1
    # The last paragraph of each chunk is repeated at the start of the next one.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.text.startswith(previous.text.split("\n\n")[-1])


def test_long_paragraph_is_split_at_sentences_and_hard_split_as_last_resort():
    sentences = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = chunk_markdown("d.md", sentences, max_chars=200, overlap_chars=50)
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)

    unbroken = chunk_markdown("d.md", "y" * 450, max_chars=200, overlap_chars=50)
    assert [len(c.text) for c in unbroken] == [200, 200, 50]


def test_rejects_overlap_not_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_markdown("d.md", DOC, max_chars=100, overlap_chars=100)


def test_sample_docs_chunk_cleanly():
    for doc in load_documents(DEFAULT_DOCS_DIR):
        chunks = chunk_markdown(doc.name, doc.text)
        assert chunks, doc.name
        assert all(0 < len(c.text) <= 1200 for c in chunks)
