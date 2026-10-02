from __future__ import annotations

import pytest
from conftest import grounding

from file_search.engine import FileSearchEngine, insert_citations, sources_from_grounding


@pytest.fixture
def engine(isolated_data_dir, fake_client) -> FileSearchEngine:
    return FileSearchEngine(client=fake_client)


class TestIndexing:
    def test_creates_store_and_uploads(self, engine, fake_client, make_docs):
        report = engine.index(make_docs({"a.md": "alpha", "b.md": "beta"}))
        assert report.added == ["a.md", "b.md"]
        assert fake_client.uploads == ["a.md", "b.md"]
        status = engine.status()
        assert status["indexed"] and status["active_documents"] == 2

    def test_incremental_sync_replaces_changed_and_deletes_removed(self, engine, fake_client, make_docs):
        engine.index(make_docs({"a.md": "alpha", "b.md": "beta"}))
        assert engine.index(make_docs({"a.md": "alpha", "b.md": "beta"})).unchanged == ["a.md", "b.md"]
        assert fake_client.uploads == ["a.md", "b.md"]

        docs = [d for d in make_docs({"a.md": "alpha v2"}) if d.name == "a.md"]
        report = engine.index(docs)
        assert (report.updated, report.removed) == (["a.md"], ["b.md"])
        assert fake_client.uploads == ["a.md", "b.md", "a.md"]
        assert len(fake_client.deleted_documents) == 2
        assert engine.status()["documents"] == ["a.md"]

    def test_recreates_store_deleted_remotely(self, engine, fake_client, make_docs):
        engine.index(make_docs({"a.md": "alpha"}))
        fake_client.stores.clear()  # e.g. deleted in AI Studio
        report = engine.index(make_docs({"a.md": "alpha"}))
        assert report.added == ["a.md"]
        assert engine.status()["store"] == "fileSearchStores/fake1"

    def test_failed_operation_raises(self, engine, fake_client, make_docs):
        upload = fake_client.file_search_stores.upload_to_file_search_store

        def failing(**kwargs):
            operation = upload(**kwargs)
            operation.error = {"message": "unsupported file"}
            return operation

        fake_client.file_search_stores.upload_to_file_search_store = failing
        with pytest.raises(RuntimeError, match="unsupported file"):
            engine.index(make_docs({"a.md": "alpha"}))

    def test_reset_deletes_store_and_state(self, engine, fake_client, make_docs):
        engine.index(make_docs({"a.md": "alpha"}))
        engine.reset()
        assert fake_client.stores == {}
        assert engine.status()["indexed"] is False


class TestAsk:
    def test_requires_index(self, engine):
        with pytest.raises(ValueError, match="empty"):
            engine.ask("anything")

    def test_answer_with_sources_and_inline_citations(self, engine, fake_client, make_docs):
        engine.index(make_docs({"a.md": "alpha"}))
        fake_client.answer_text = "Hotels cost 230 EUR. Flights are economy."
        fake_client.grounding = grounding(
            chunks=[("travel.md", "London: 230 EUR"), ("travel.md", "Economy class"), ("hr.md", "unused")],
            supports=[(0, 20, [0]), (21, 41, [1])],
        )
        answer = engine.ask("hotel and flights?")
        assert answer.text == "Hotels cost 230 EUR. [1] Flights are economy. [2]"
        assert [(s.number, s.document, s.cited) for s in answer.sources] == [
            (1, "travel.md", True),
            (2, "travel.md", True),
            (3, "hr.md", False),
        ]

    def test_answer_without_grounding(self, engine, fake_client, make_docs):
        engine.index(make_docs({"a.md": "alpha"}))
        fake_client.answer_text = "I couldn't find that in the documents."
        answer = engine.ask("capital of Peru?")
        assert answer.sources == [] and answer.text == fake_client.answer_text


def test_insert_citations_uses_utf8_byte_offsets():
    text = "Core hours 10:00–14:00. Sauna on Fridays."
    first_end = len("Core hours 10:00–14:00.".encode("utf-8"))  # "–" is 3 bytes
    metadata = grounding([("a", "x"), ("b", "y")], [(0, first_end, [0, 1]), (first_end + 1, len(text.encode()), [1])])
    assert insert_citations(text, metadata) == "Core hours 10:00–14:00. [1, 2] Sauna on Fridays. [2]"


def test_grounding_helpers_tolerate_missing_metadata():
    assert sources_from_grounding(None) == []
    assert insert_citations("text", None) == "text"
