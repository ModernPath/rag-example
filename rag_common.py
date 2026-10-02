"""Shared pieces for both RAG engines: config, documents, and result types.

Both engines (scratch/ and file_search/) take the same Documents in and return the
same Answer out, so the CLI, UI, and tests can treat them interchangeably.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

RAG_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = RAG_DIR / "sample_docs"
SUPPORTED_SUFFIXES = (".md", ".txt")

API_KEY_VARS = ("GOOGLE_AI_STUDIO_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def generation_model() -> str:
    return os.environ.get("RAG_MODEL", "gemini-3.8-flash")


def embedding_model() -> str:
    return os.environ.get("RAG_EMBEDDING_MODEL", "gemini-embedding-2")


def chunking_model() -> str:
    """Small, fast model used by the semantic chunker (scratch-semantic engine)."""
    return os.environ.get("RAG_CHUNKING_MODEL", "gemini-3.5-flash-lite")


def data_dir() -> Path:
    """Where indexes and store state live. Override with RAG_DATA_DIR."""
    return Path(os.environ.get("RAG_DATA_DIR", RAG_DIR / "data"))


def load_environment() -> None:
    """Load .env / .env.local from this folder and its parents (closest wins)."""
    from dotenv import load_dotenv

    for directory in reversed([RAG_DIR, *RAG_DIR.parents]):
        for name in (".env", ".env.local"):
            path = directory / name
            if path.is_file():
                load_dotenv(path, override=True)


def get_client():
    """Gemini client. Keep the returned object referenced while you use it:
    the SDK closes its HTTP connection when the client is garbage-collected."""
    api_key = next((os.environ[v] for v in API_KEY_VARS if os.environ.get(v)), None)
    if not api_key:
        raise RuntimeError(f"No Gemini API key set (tried {', '.join(API_KEY_VARS)}).")
    from google import genai

    return genai.Client(api_key=api_key)


def generate_text(prompt: str, *, system: str) -> str:
    """Plain text generation (no tools) with the configured model."""
    from google.genai import types

    client = get_client()
    response = client.models.generate_content(
        model=generation_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            # No local tools to call; also silences google-genai 2.x's AFC warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return (response.text or "").strip()


def api_errors() -> tuple:
    """Exception types to report as friendly errors (Gemini API errors included when installed)."""
    try:
        from google.genai.errors import APIError
    except ImportError:
        return (ValueError, RuntimeError)
    return (ValueError, RuntimeError, APIError)


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Document:
    name: str  # file name, used as the stable document id
    path: Path
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def load_documents(docs_dir: Path = DEFAULT_DOCS_DIR) -> List[Document]:
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise ValueError(f"Documents folder not found: {docs_dir}")
    return [
        Document(name=path.name, path=path, text=path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.iterdir())
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file()
    ]


@dataclass
class SyncPlan:
    """Which documents changed since the last index run, based on content hashes."""

    new: List[Document] = field(default_factory=list)
    changed: List[Document] = field(default_factory=list)
    unchanged: List[Document] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def to_index(self) -> List[Document]:
        return [*self.new, *self.changed]


def plan_sync(documents: Sequence[Document], indexed_hashes: Dict[str, str]) -> SyncPlan:
    plan = SyncPlan()
    for doc in documents:
        previous = indexed_hashes.get(doc.name)
        if previous is None:
            plan.new.append(doc)
        elif previous != doc.sha256:
            plan.changed.append(doc)
        else:
            plan.unchanged.append(doc)
    current = {doc.name for doc in documents}
    plan.removed = sorted(name for name in indexed_hashes if name not in current)
    return plan


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class Source:
    """One retrieved passage that the answer may cite as [number]."""

    number: int
    document: str
    title: str
    text: str
    score: Optional[float] = None  # similarity; File Search does not expose it
    cited: bool = False


@dataclass
class Answer:
    engine: str
    question: str
    text: str
    sources: List[Source]
    seconds: float


@dataclass
class IndexReport:
    engine: str
    added: List[str]
    updated: List[str]
    removed: List[str]
    unchanged: List[str]
    seconds: float

    def summary(self) -> str:
        return (
            f"{self.engine}: {len(self.added)} added, {len(self.updated)} updated, "
            f"{len(self.removed)} removed, {len(self.unchanged)} unchanged ({self.seconds:.1f}s)"
        )


class RagEngine(Protocol):
    name: str

    def index(self, documents: Sequence[Document]) -> IndexReport: ...

    def ask(self, question: str, *, top_k: int = 5) -> Answer: ...

    def status(self) -> Dict[str, object]: ...

    def reset(self) -> None: ...


# --------------------------------------------------------------------------- #
# Prompting (shared so both engines answer in the same style)
# --------------------------------------------------------------------------- #

ANSWER_RULES = """\
Answer the question using only the provided documents.
- If the documents do not contain the answer, say "I couldn't find that in the documents."
- Be concise: one short paragraph or a few "- " bullet points, in plain text (no Markdown bold or headings).
- Quote exact numbers, names, and limits as written in the documents.
"""

NOT_FOUND_REPLY = "I couldn't find that in the documents."
