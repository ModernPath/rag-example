"""Base language: one language for the index, questions translated into it before retrieval.

Why a base language at all? Embeddings are multilingual, but not perfectly so: a Finnish
question against English passages scores lower than the same question in English, so the
score threshold and top-k behave differently per language. And the semantic chunker writes
chunk titles, which are embedded together with the text (`title: ... | text: ...`), so titles
in mixed languages would scatter the index. Fixing one base language keeps every stage
comparable:

    index:  documents (any language) ─► chunks with titles in the BASE language ─► embed
    ask:    question (any language) ─► detect + translate into the BASE language
                                   ─► embed / retrieve / prompt in the BASE language
                                   ─► the model replies in the language of the ORIGINAL question

The chunk text itself is never translated: it stays verbatim, so answers quote the source
as written. Only titles and questions are normalized.

The translation is one small structured-output call (`RAG_TRANSLATION_MODEL`). It returns
the detected language and the question in the base language; if the question is already in
the base language we keep the original wording rather than the model's rephrasing. Set
`RAG_TRANSLATE_QUERIES=0` to skip the call entirely (questions are then assumed to be in
the base language).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

from rag_common import get_client

DEFAULT_BASE_LANGUAGE = "en"
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")

# Enough for prompts; an unknown code is passed through as-is (the model understands codes).
LANGUAGE_NAMES = {
    "en": "English", "fi": "Finnish", "sv": "Swedish", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "da": "Danish", "no": "Norwegian", "et": "Estonian",
    "pl": "Polish", "ru": "Russian", "uk": "Ukrainian", "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "tr": "Turkish",
}


def base_language() -> str:
    """ISO 639-1 code of the index language. Override with RAG_BASE_LANGUAGE."""
    value = os.environ.get("RAG_BASE_LANGUAGE", DEFAULT_BASE_LANGUAGE).strip().lower()
    if not _LANGUAGE_CODE.match(value):
        raise ValueError(f"RAG_BASE_LANGUAGE must be an ISO 639-1 code such as 'en' or 'fi', not {value!r}.")
    return value


def translation_model() -> str:
    """Small, fast model for detecting the question language and translating it."""
    return os.environ.get("RAG_TRANSLATION_MODEL", "gemini-3.5-flash-lite")


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


@dataclass(frozen=True)
class Query:
    """A question as asked, and as the engines see it (in the base language)."""

    original: str
    language: str  # detected language of the original question
    text: str  # the question in the base language
    base_language: str

    @property
    def translated(self) -> bool:
        return self.text != self.original


Translator = Callable[[str], Query]


def reply_language_rule(query: Query) -> str:
    """System prompt line so the answer comes back in the language the user asked in."""
    return f"Reply in {language_name(query.language)}, the language of the original question."


def passthrough(question: str) -> Query:
    """No translation: treat the question as already written in the base language."""
    base = base_language()
    return Query(original=question, language=base, text=question, base_language=base)


def default_translator() -> Translator:
    if os.environ.get("RAG_TRANSLATE_QUERIES", "1").lower() in ("0", "false", "off", "no"):
        return passthrough
    return GeminiTranslator()


# --------------------------------------------------------------------------- #
# Gemini translation (structured output), with a pure parser for tests
# --------------------------------------------------------------------------- #

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string"}, "text": {"type": "string"}},
    "required": ["language", "text"],
}


def translation_instructions(base: str) -> str:
    name = language_name(base)
    return (
        f"You prepare search questions for a {name} document index.\n"
        "- Detect the language of the question and return it as `language`, an ISO 639-1 code.\n"
        f"- Return the question translated into {name} as `text`. Keep names, product names,\n"
        "  numbers, and units exactly as written. Translate the question only; never answer it.\n"
        f"- If the question is already in {name}, return it unchanged."
    )


def parse_translation(payload: str, *, original: str, base: str) -> Query:
    """Validate the model's JSON. Pure function: the tests cover it without an API key."""
    try:
        data = json.loads(payload)
        language = str(data["language"]).strip().lower()
        text = str(data["text"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Translation returned invalid JSON: {exc}") from exc
    if not _LANGUAGE_CODE.match(language):
        raise ValueError(f"Translation returned an invalid language code: {language!r}")
    if not text:
        raise ValueError("Translation returned an empty question.")
    if language == base:
        text = original  # already in the base language: keep the user's wording
    return Query(original=original, language=language, text=text, base_language=base)


class GeminiTranslator:
    def __init__(self, base: Optional[str] = None, model: Optional[str] = None) -> None:
        self.base = base or base_language()
        self.model = model or translation_model()

    def __call__(self, question: str) -> Query:
        from google.genai import types

        client = get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=translation_instructions(self.base),
                response_mime_type="application/json",
                response_json_schema=RESPONSE_SCHEMA,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return parse_translation(response.text or "", original=question, base=self.base)
