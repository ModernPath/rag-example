from __future__ import annotations

import numpy as np
import pytest

from rag_common import NOT_FOUND_REPLY, Source, plan_sync
from scratch.chunking import Chunk
from scratch.engine import ScratchEngine, build_prompt, mark_cited
from scratch.vector_store import ChromaVectorStore

HANDBOOK = "# Handbook\n\n## Vacation\n\nEveryone gets 25 vacation days.\n\n## Sauna\n\nThe sauna opens Friday."
TRAVEL = "# Travel\n\n## Hotels\n\nLondon hotel limit is 230 EUR."


@pytest.fixture
def engine(isolated_data_dir, fake_embedder, fake_generate) -> ScratchEngine:
    return ScratchEngine(embedder=fake_embedder, generate=fake_generate, min_score=0.1)


class TestChromaVectorStore:
    def _store(self, path) -> ChromaVectorStore:
        store = ChromaVectorStore(path, embedding_model="m", dimensions=2)
        store.add_document("h1", [Chunk("a#0", "a", "A", "x")], np.array([[1, 0]], dtype=np.float32))
        store.add_document("h2", [Chunk("b#0", "b", "B", "y")], np.array([[0, 1]], dtype=np.float32))
        return store

    def test_search_ranks_by_cosine_and_applies_threshold(self, tmp_path):
        store = self._store(tmp_path / "db")
        hits = store.search(np.array([0.8, 0.6], dtype=np.float32), top_k=5)
        assert [(c.id, c.title, c.text, round(s, 2)) for c, s in hits] == [("a#0", "A", "x", 0.8), ("b#0", "B", "y", 0.6)]
        assert [c.id for c, _ in store.search(np.array([0.8, 0.6]), min_score=0.7)] == ["a#0"]

    def test_hashes_remove_and_persistence(self, tmp_path):
        store = self._store(tmp_path / "db")
        assert store.document_hashes() == {"a": "h1", "b": "h2"}
        store.remove_document("a")
        reopened = ChromaVectorStore(tmp_path / "db", embedding_model="m", dimensions=2)
        assert len(reopened) == 1 and reopened.document_hashes() == {"b": "h2"}

    def test_other_embedder_starts_fresh(self, tmp_path):
        self._store(tmp_path / "db")
        assert len(ChromaVectorStore(tmp_path / "db", embedding_model="other", dimensions=2)) == 0

    def test_rejects_mismatched_input(self, tmp_path):
        store = ChromaVectorStore(tmp_path / "db", embedding_model="m", dimensions=2)
        with pytest.raises(ValueError):
            store.add_document("h", [Chunk("a#0", "a", "A", "x")], np.zeros((2, 2)))
        with pytest.raises(ValueError):
            store.add_document("h", [Chunk("a#0", "a", "A", "x")], np.zeros((1, 3)))

    def test_empty_store_search(self, tmp_path):
        assert ChromaVectorStore(tmp_path / "db", embedding_model="m", dimensions=2).search(np.ones(2)) == []


class TestScratchEngine:
    def test_index_and_retrieve(self, engine, make_docs):
        report = engine.index(make_docs({"handbook.md": HANDBOOK, "travel.md": TRAVEL}))
        assert report.added == ["handbook.md", "travel.md"]

        top = engine.retrieve("London hotel limit", top_k=1)[0]
        assert (top.document, top.title) == ("travel.md", "Travel > Hotels")
        assert top.score > 0.5

    def test_incremental_sync(self, engine, make_docs, fake_embedder):
        engine.index(make_docs({"handbook.md": HANDBOOK, "travel.md": TRAVEL}))
        calls = fake_embedder.document_calls

        report = engine.index(make_docs({"handbook.md": HANDBOOK, "travel.md": TRAVEL}))
        assert report.unchanged == ["handbook.md", "travel.md"]
        assert fake_embedder.document_calls == calls  # nothing re-embedded

        docs = make_docs({"handbook.md": HANDBOOK.replace("25", "30")})
        report = engine.index(docs)
        assert (report.updated, report.removed) == (["handbook.md"], ["travel.md"])
        assert engine.status()["documents"] == ["handbook.md"]
        assert "30 vacation days" in engine.retrieve("vacation days", top_k=1)[0].text

    def test_ask_builds_grounded_prompt_and_marks_citations(self, engine, make_docs, fake_generate):
        engine.index(make_docs({"handbook.md": HANDBOOK}))
        answer = engine.ask("How many vacation days?", top_k=2)
        assert answer.text == "Answer from the documents [1]."
        assert [s.cited for s in answer.sources] == [True, False]
        assert "Question: How many vacation days?" in fake_generate.prompts[0]
        assert "[1] (handbook.md — Handbook > Vacation)" in fake_generate.prompts[0]

    def test_ask_skips_llm_when_nothing_relevant(self, make_docs, fake_embedder, fake_generate):
        engine = ScratchEngine(embedder=fake_embedder, generate=fake_generate, min_score=0.99)
        engine.index(make_docs({"handbook.md": HANDBOOK}))
        assert engine.ask("pancake recipe").text == NOT_FOUND_REPLY
        assert fake_generate.prompts == []

    def test_empty_index_raises(self, engine):
        with pytest.raises(ValueError, match="empty"):
            engine.retrieve("anything")

    def test_status_and_reset(self, engine, make_docs):
        assert engine.status()["indexed"] is False
        engine.index(make_docs({"handbook.md": HANDBOOK}))
        status = engine.status()
        assert status["indexed"] and status["chunks"] == 2 and status["embedding_model"] == "fake-embedder"
        assert status["chunking"] == "markdown"
        engine.reset()
        assert engine.status()["indexed"] is False

    def test_changing_embedder_rebuilds_index(self, engine, make_docs, fake_embedder):
        engine.index(make_docs({"handbook.md": HANDBOOK}))
        fake_embedder.model = "another-model"
        report = engine.index(make_docs({"handbook.md": HANDBOOK}))
        assert report.added == ["handbook.md"]  # old vectors were incompatible


def test_plan_sync(make_docs):
    docs = make_docs({"a.md": "same", "b.md": "new text", "c.md": "brand new"})
    plan = plan_sync(docs, {"a.md": docs[0].sha256, "b.md": "old-hash", "gone.md": "x"})
    assert [d.name for d in plan.unchanged] == ["a.md"]
    assert [d.name for d in plan.changed] == ["b.md"]
    assert [d.name for d in plan.new] == ["c.md"]
    assert plan.removed == ["gone.md"]


def test_mark_cited_handles_grouped_citations():
    sources = [Source(n, "d", "t", "x") for n in (1, 2, 3)]
    mark_cited("Yes [1, 3].", sources)
    assert [s.cited for s in sources] == [True, False, True]


def test_build_prompt_numbers_passages():
    prompt = build_prompt("Q?", [Source(1, "a.md", "A > B", "text")])
    assert prompt == "Documents:\n\n[1] (a.md — A > B)\ntext\n\nQuestion: Q?"
