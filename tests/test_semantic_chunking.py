from __future__ import annotations

from scratch.engine import semantic_engine
from scratch.semantic_chunking import Boundary, SemanticChunker, chunks_from_boundaries, split_sentences

TEXT = """Acme Intranet | Home | Log out

Book flights through the portal. Economy class for short trips. Hotels: London 230 EUR per night.
Paris 210 EUR per night.

Submit receipts within 30 days.
(c) Acme | Privacy"""
# Sentences: [1] nav, [2] flights, [3] economy, [4] London, [5] Paris, [6] receipts, [7] footer


class RecordingSegmenter:
    """Stands in for the LLM: returns fixed boundaries and records what it was sent."""

    def __init__(self, boundaries):
        self.boundaries = boundaries
        self.calls = []

    def __call__(self, document, numbered):
        self.calls.append((document, numbered))
        return self.boundaries


TOPICS = [
    Boundary(1, "Navigation", skip=True),
    Boundary(2, "Travel - Flights"),
    Boundary(4, "Travel - Hotels"),
    Boundary(6, "Travel - Receipts"),
    Boundary(7, "Footer", skip=True),
]


def test_split_sentences_keeps_offsets_into_original_text():
    sentences = split_sentences(TEXT)
    assert len(sentences) == 7
    assert all(TEXT[offset:].startswith(sentence) for offset, sentence in sentences)
    assert sentences[3][1] == "Hotels: London 230 EUR per night."


def test_cuts_original_text_mid_paragraph_and_drops_boilerplate():
    chunks = SemanticChunker(segment=RecordingSegmenter(TOPICS))("travel.txt", TEXT)

    assert [(c.id, c.title) for c in chunks] == [
        ("travel.txt#0", "Travel - Flights"),
        ("travel.txt#1", "Travel - Hotels"),
        ("travel.txt#2", "Travel - Receipts"),
    ]
    assert chunks[0].text == "Book flights through the portal. Economy class for short trips."
    assert chunks[1].text == "Hotels: London 230 EUR per night.\nParis 210 EUR per night."  # verbatim
    assert not any("Intranet" in c.text or "Privacy" in c.text for c in chunks)


def test_model_sees_numbered_sentences():
    segment = RecordingSegmenter([Boundary(1, "All")])
    SemanticChunker(segment=segment)("travel.txt", TEXT)
    document, numbered = segment.calls[0]
    assert document == "travel.txt"
    assert numbered.splitlines()[:3] == [
        "[1] Acme Intranet | Home | Log out",
        "[2] Book flights through the portal.",
        "[3] Economy class for short trips.",
    ]


def test_sanitizes_bad_boundaries():
    # Unsorted, duplicate, out of range, and missing sentence 1.
    boundaries = [Boundary(6, "Receipts"), Boundary(4, "Hotels"), Boundary(4, "Dup"), Boundary(99, "Nope"), Boundary(0, "Zero")]
    chunks = chunks_from_boundaries("t", TEXT, boundaries)
    assert [c.title for c in chunks] == ["t", "Hotels", "Receipts"]
    assert "".join(c.text for c in chunks).replace("\n", "").replace(" ", "") == TEXT.replace("\n", "").replace(" ", "")


def test_no_boundaries_gives_one_chunk_and_empty_text_gives_none():
    assert [c.title for c in chunks_from_boundaries("t", TEXT, [])] == ["t"]
    assert SemanticChunker(segment=RecordingSegmenter([]))("t", "  \n\n") == []


def test_oversized_chunk_is_split_by_paragraphs():
    chunks = chunks_from_boundaries("t", TEXT, [Boundary(1, "Everything")], max_chars=80)
    assert len(chunks) > 1 and all(len(c.text) <= 80 for c in chunks)
    assert {c.title for c in chunks} == {"Everything"}


def test_semantic_engine_indexes_with_its_own_chunker(isolated_data_dir, make_docs, fake_embedder, fake_generate):
    segment = RecordingSegmenter(TOPICS)
    engine = semantic_engine(chunker=SemanticChunker(segment=segment), embedder=fake_embedder, generate=fake_generate, min_score=0.1)
    engine.index(make_docs({"travel.txt": TEXT}))

    status = engine.status()
    assert (status["engine"], status["chunking"], status["chunks"]) == ("scratch-semantic", "semantic", 3)
    assert status["location"] == str(isolated_data_dir / "scratch-semantic")  # separate index from "scratch"
    assert engine.retrieve("London hotel per night", top_k=1)[0].title == "Travel - Hotels"

    engine.index(make_docs({"travel.txt": TEXT}))
    assert len(segment.calls) == 1  # unchanged document: no second LLM call
