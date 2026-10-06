from __future__ import annotations

import os
import subprocess
import sys

import pytest
from conftest import RAG_DIR, FakeTranslator

import rag
from rag_common import Answer, IndexReport, Source, load_documents
from translation import Query
from ui.app import create_app


class FakeEngine:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.indexed_docs = 0

    def index(self, documents):
        self.indexed_docs = len(documents)
        return IndexReport(self.name, [d.name for d in documents], [], [], [], 0.1)

    def ask(self, question, *, top_k=5):
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        self.asked = question
        query = question if isinstance(question, Query) else Query(question, "en", question, "en")
        source = Source(1, "doc.md", "Doc > Section", "passage text", score=0.8, cited=True)
        return Answer(self.name, query.original, f"{self.name} says hi [1]", [source], 0.2, query=query)

    def status(self):
        return {"engine": self.name, "indexed": self.indexed_docs > 0, "documents": ["doc.md"] * self.indexed_docs}

    def reset(self):
        self.indexed_docs = 0


@pytest.fixture
def engines():
    return {"scratch": FakeEngine("scratch"), "file-search": FakeEngine("file-search", fail=True)}


@pytest.fixture
def translator():
    return FakeTranslator({"Montako lomapäivää?": ("fi", "How many vacation days?")})


@pytest.fixture
def client(engines, translator):
    app = create_app(engines, translator=translator)
    app.config["TESTING"] = True
    return app.test_client()


def test_dashboard_shows_engine_status(client):
    page = client.get("/").get_data(as_text=True)
    assert page.count("Not indexed yet.") == 2


def test_index_button_indexes_sample_docs(client, engines):
    page = client.post("/index/scratch", follow_redirects=True).get_data(as_text=True)
    count = len(load_documents())
    assert f"scratch: {count} added" in page
    assert engines["scratch"].indexed_docs == count
    assert "Unknown engine" in client.post("/index/nope", follow_redirects=True).get_data(as_text=True)


def test_ask_compares_engines_and_isolates_failures(client):
    page = client.get("/?q=hello").get_data(as_text=True)
    assert "scratch says hi [1]" in page
    assert "file-search is down" in page
    assert "[1] Doc &gt; Section" in page


def test_ask_selected_engine_only(client):
    page = client.get("/?q=hello&engine=scratch").get_data(as_text=True)
    assert "scratch says hi" in page
    assert "file-search is down" not in page


def test_ui_translates_once_and_shows_the_translation(client, engines, translator):
    page = client.get("/?q=Montako+lomapäivää%3F&engine=scratch").get_data(as_text=True)
    assert translator.calls == ["Montako lomapäivää?"]  # one call shared by all engines
    assert engines["scratch"].asked == Query("Montako lomapäivää?", "fi", "How many vacation days?", "en")
    assert "Translated from Finnish to English" in page and "How many vacation days?" in page


def test_ui_shows_no_translation_line_for_base_language_questions(client):
    page = client.get("/?q=hello&engine=scratch").get_data(as_text=True)
    assert "Translated from" not in page
    assert "Base language: <code>en</code>" in page


def test_cli_ask_and_search_print_the_translation(monkeypatch, capsys, translator):
    engine = FakeEngine("scratch")
    engine.retrieve = lambda question, top_k=5: [Source(1, "doc.md", "Doc", "passage", score=0.7)]
    monkeypatch.setattr(rag, "make_engines", lambda choice: [engine])
    monkeypatch.setattr(rag, "make_translator", lambda: translator)

    assert rag.main(["ask", "Montako lomapäivää?"]) == 0
    out = capsys.readouterr().out
    assert "Translated (fi → en): How many vacation days?" in out and "scratch says hi [1]" in out
    assert engine.asked.text == "How many vacation days?"

    assert rag.main(["search", "Montako lomapäivää?"]) == 0
    assert "Translated (fi → en): How many vacation days?" in capsys.readouterr().out

    assert rag.main(["ask", "hello"]) == 0
    assert "Translated" not in capsys.readouterr().out


def test_cli_help_and_empty_index():
    # The CLI translates the question before the engine reports an empty index; keep that offline.
    env = {**os.environ, "RAG_TRANSLATE_QUERIES": "0"}
    run = lambda *args: subprocess.run(  # noqa: E731
        [sys.executable, str(RAG_DIR / "rag.py"), *args], capture_output=True, text=True, timeout=60, env=env
    )
    assert "vs Gemini File Search" in run("--help").stdout

    status = run("status", "--engine", "scratch")
    assert status.returncode == 0 and "'indexed': False" in status.stdout

    search = run("search", "anything")
    assert search.returncode == 1 and "index is empty" in search.stderr

    search = run("search", "anything", "--engine", "scratch-semantic")
    assert search.returncode == 1 and "index is empty" in search.stderr


def test_cli_previews_markdown_chunks():
    result = subprocess.run(
        [sys.executable, str(RAG_DIR / "rag.py"), "chunks", "travel-and-expenses.md"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0
    assert "[travel-and-expenses.md#0]" in result.stdout and "Travel and Expense Policy > Hotels" in result.stdout
