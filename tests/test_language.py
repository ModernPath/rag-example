"""Base language: chunk titles are written in it, and questions are translated into it
before retrieval and generation. Offline: the translator is a fake."""

from __future__ import annotations

import json

import pytest
from conftest import FakeTranslator, grounding

from file_search.engine import FileSearchEngine
from rag_common import NOT_FOUND_REPLY
from scratch.engine import ScratchEngine
from scratch.semantic_chunking import SemanticChunker, segment_instructions
from translation import Query, base_language, language_name, parse_translation, passthrough

HANDBOOK = "# Handbook\n\n## Vacation\n\nEveryone gets 25 vacation days.\n\n## Sauna\n\nThe sauna opens Friday."


# -- configuration ---------------------------------------------------------- #


def test_base_language_defaults_to_english_and_reads_env(monkeypatch):
    monkeypatch.delenv("RAG_BASE_LANGUAGE", raising=False)
    assert base_language() == "en"
    monkeypatch.setenv("RAG_BASE_LANGUAGE", "FI")
    assert base_language() == "fi"  # normalized to a lowercase ISO 639-1 code


def test_base_language_rejects_free_text(monkeypatch):
    monkeypatch.setenv("RAG_BASE_LANGUAGE", "Finnish please")
    with pytest.raises(ValueError, match="RAG_BASE_LANGUAGE"):
        base_language()


def test_language_name_for_prompts():
    assert language_name("fi") == "Finnish"
    assert language_name("en") == "English"
    assert language_name("xx") == "xx"  # unknown code: pass through rather than guess


# -- translation ------------------------------------------------------------ #


def test_parse_translation_validates_model_output():
    query = parse_translation(json.dumps({"language": "fi", "text": "How many vacation days?"}),
                              original="Montako lomapäivää?", base="en")
    assert query == Query(original="Montako lomapäivää?", language="fi", text="How many vacation days?", base_language="en")
    assert query.translated


def test_parse_translation_keeps_original_when_already_in_base_language():
    # Never let the model "rephrase" a question that needs no translation.
    query = parse_translation(json.dumps({"language": "en", "text": "Vacation days, how many?"}),
                              original="How many vacation days?", base="en")
    assert query.text == "How many vacation days?"
    assert not query.translated


@pytest.mark.parametrize("payload", ['{"language": "fi"}', '{"language": "Finnish", "text": "x"}', '{"language": "fi", "text": ""}', "nope"])
def test_parse_translation_rejects_invalid_output(payload):
    with pytest.raises(ValueError, match="Translation"):
        parse_translation(payload, original="q", base="en")


def test_passthrough_marks_question_as_base_language(monkeypatch):
    monkeypatch.setenv("RAG_BASE_LANGUAGE", "en")
    assert passthrough("Hello") == Query("Hello", "en", "Hello", "en")


# -- scratch engine --------------------------------------------------------- #


@pytest.fixture
def translator():
    return FakeTranslator({"Montako lomapäivää saan?": ("fi", "How many vacation days do I get?")})


@pytest.fixture
def engine(isolated_data_dir, fake_embedder, fake_generate, translator, make_docs) -> ScratchEngine:
    engine = ScratchEngine(embedder=fake_embedder, generate=fake_generate, translator=translator, min_score=0.1)
    engine.index(make_docs({"handbook.md": HANDBOOK}))
    return engine


def test_ask_retrieves_and_prompts_in_base_language_but_replies_in_the_question_language(engine, fake_generate):
    answer = engine.ask("Montako lomapäivää saan?")
    assert answer.query == Query("Montako lomapäivää saan?", "fi", "How many vacation days do I get?", "en")
    assert "Question: How many vacation days do I get?" in fake_generate.prompts[0]
    assert "25 vacation days" in answer.sources[0].text  # retrieved with the English query
    assert "Reply in Finnish" in fake_generate.systems[0]
    assert answer.question == "Montako lomapäivää saan?"


def test_ask_accepts_a_pretranslated_query_without_calling_the_translator(engine, translator):
    query = Query("Montako lomapäivää saan?", "fi", "How many vacation days do I get?", "en")
    answer = engine.ask(query)
    assert answer.query is query
    assert translator.calls == []


def test_search_translates_too(engine, translator):
    assert "25 vacation days" in engine.retrieve("Montako lomapäivää saan?", top_k=1)[0].text
    assert translator.calls == ["Montako lomapäivää saan?"]


def test_not_found_reply_is_given_in_the_question_language(engine, fake_generate):
    engine.min_score = 0.99
    answer = engine.ask("Montako lomapäivää saan?")
    assert answer.text == NOT_FOUND_REPLY
    assert fake_generate.prompts == []  # still no LLM call for the answer
    assert answer.query.language == "fi"


def test_status_reports_base_language(engine):
    assert engine.status()["base_language"] == "en"


# -- semantic chunking ------------------------------------------------------ #


def test_semantic_chunker_titles_in_base_language(monkeypatch):
    monkeypatch.setenv("RAG_BASE_LANGUAGE", "fi")
    assert SemanticChunker().base_language == "fi"
    instructions = segment_instructions("fi")
    assert "Finnish" in instructions
    assert "Write every title in Finnish" in instructions
    assert "Write every title in English" in segment_instructions("en")  # the model's default is the document's language


# -- file search engine ----------------------------------------------------- #


def test_file_search_translates_question_and_asks_for_reply_language(isolated_data_dir, fake_client, make_docs, translator):
    engine = FileSearchEngine(client=fake_client, translator=translator)
    engine.index(make_docs({"handbook.md": HANDBOOK}))
    fake_client.answer_text = "Saat 25 lomapäivää."
    fake_client.grounding = grounding([("handbook.md", "Everyone gets 25 vacation days.")], [(0, 21, [0])])
    answer = engine.ask("Montako lomapäivää saan?")
    assert fake_client.generate_calls[0]["contents"] == "How many vacation days do I get?"
    assert "Reply in Finnish" in fake_client.generate_calls[0]["config"].system_instruction
    assert answer.query.translated
    assert engine.status()["base_language"] == "en"
