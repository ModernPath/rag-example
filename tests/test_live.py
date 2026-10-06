"""End-to-end against the real Gemini API. Opt in: RAG_LIVE_TESTS=1 pytest tests/test_live.py

Creates (and deletes) a temporary File Search store; costs a few cents of embeddings.
"""

from __future__ import annotations

import os

import pytest

from rag_common import DEFAULT_DOCS_DIR, load_documents, load_environment

pytestmark = pytest.mark.skipif(os.environ.get("RAG_LIVE_TESTS") != "1", reason="set RAG_LIVE_TESTS=1")

QUESTIONS = [
    ("How many vacation days do I get after four years of employment?", "30"),
    ("What is the maximum hotel price per night in London?", "230"),
    ("How fast must I report a lost laptop?", "hour"),
]


@pytest.fixture(scope="module")
def engines():
    load_environment()
    from file_search.engine import FileSearchEngine
    from scratch.engine import ScratchEngine, semantic_engine

    built = [ScratchEngine(), semantic_engine(), FileSearchEngine()]
    for engine in built:
        engine.index(load_documents())
    yield built
    for engine in built:
        engine.reset()


@pytest.mark.parametrize("question, expected", QUESTIONS)
def test_engines_answer_from_documents(engines, question, expected):
    for engine in engines:
        answer = engine.ask(question)
        assert expected in answer.text, f"{engine.name}: {answer.text}"
        assert any(s.cited for s in answer.sources), f"{engine.name} cited nothing"


def test_engines_decline_out_of_scope_questions(engines):
    for engine in engines:
        assert "couldn't find" in engine.ask("What is the capital of Peru?").text.lower()


def test_semantic_chunker_keeps_text_verbatim():
    load_environment()
    from scratch.semantic_chunking import SemanticChunker

    text = (DEFAULT_DOCS_DIR / "travel-and-expenses.md").read_text(encoding="utf-8")
    chunks = SemanticChunker()("travel-and-expenses.md", text)
    assert len(chunks) >= 3
    assert all(c.text in text and c.title for c in chunks)
    assert any("230" in c.text and "hotel" in c.title.lower() for c in chunks)


def test_questions_in_another_language_are_translated_and_answered_in_kind(engines):
    for engine in engines:
        answer = engine.ask("Montako lomapäivää saan neljän vuoden jälkeen?")
        assert answer.query.language == "fi" and answer.query.translated, answer.query
        assert "vacation" in answer.query.text.lower(), answer.query
        assert "30" in answer.text, f"{engine.name}: {answer.text}"
        assert "lomap" in answer.text.lower(), f"{engine.name} did not reply in Finnish: {answer.text}"


def test_base_language_question_finds_finnish_document(engines):
    for engine in engines:
        answer = engine.ask("How much is the bicycle benefit per year?")
        assert not answer.query.translated
        assert "1 200" in answer.text or "1200" in answer.text or "1,200" in answer.text, f"{engine.name}: {answer.text}"
        assert any(s.document.startswith("tyosuhde-edut") and s.cited for s in answer.sources), engine.name


def test_semantic_chunker_titles_finnish_document_in_base_language():
    load_environment()
    from scratch.semantic_chunking import SemanticChunker

    text = (DEFAULT_DOCS_DIR / "tyosuhde-edut.md").read_text(encoding="utf-8")
    chunks = SemanticChunker(base="en")("tyosuhde-edut.md", text)
    assert all(c.text in text for c in chunks)
    titles = " ".join(c.title.lower() for c in chunks)
    assert "bicycle" in titles or "bike" in titles, titles
    assert "polkupyörä" not in titles, titles
