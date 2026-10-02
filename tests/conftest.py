"""Shared test fixtures: offline fakes for the embedder, the LLM, and File Search."""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pytest

RAG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAG_DIR))

from rag_common import Document  # noqa: E402
from scratch.embedder import normalize  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def make_docs(tmp_path: Path):
    """Write {name: text} into a docs folder and return loaded Documents."""

    def make(files: Dict[str, str]) -> List[Document]:
        folder = tmp_path / "docs"
        folder.mkdir(exist_ok=True)
        for name, text in files.items():
            (folder / name).write_text(text, encoding="utf-8")
        return [Document(name, folder / name, text) for name, text in sorted(files.items())]

    return make


class FakeEmbedder:
    """Hashed bag-of-words vectors: shared words => higher cosine similarity."""

    model = "fake-embedder"
    dimensions = 64

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, titles, texts) -> np.ndarray:
        self.document_calls += 1
        vectors = [self._vector(f"{t} {x}") for t, x in zip(titles, texts)]
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimensions), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimensions] += 1
        return normalize(vector)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


class FakeGenerator:
    """Records prompts and answers by citing passage [1]."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def __call__(self, prompt: str, *, system: str) -> str:
        self.prompts.append(prompt)
        return "Answer from the documents [1]."


@pytest.fixture
def fake_generate() -> FakeGenerator:
    return FakeGenerator()


class NotFound(Exception):
    code = 404


class FakeFileSearchClient:
    """Implements just the google-genai calls FileSearchEngine makes."""

    def __init__(self) -> None:
        self.stores: Dict[str, Dict[str, str]] = {}  # store name -> {document name: display name}
        self.uploads: List[str] = []
        self.deleted_documents: List[str] = []
        self.grounding = None
        self.answer_text = ""

        self.file_search_stores = SimpleNamespace(
            create=self._create,
            get=self._get,
            delete=lambda name, config=None: self.stores.pop(name, None),
            upload_to_file_search_store=self._upload,
            documents=SimpleNamespace(delete=self._delete_document),
        )
        self.operations = SimpleNamespace(get=lambda operation: operation)
        self.models = SimpleNamespace(generate_content=self._generate)

    def _create(self, config):
        name = f"fileSearchStores/fake{len(self.stores) + 1}"
        self.stores[name] = {}
        return SimpleNamespace(name=name)

    def _get(self, name):
        if name not in self.stores:
            raise NotFound(f"{name} NOT_FOUND")
        documents = self.stores[name]
        return SimpleNamespace(name=name, active_documents_count=len(documents), pending_documents_count=0, size_bytes=1)

    def _upload(self, file, file_search_store_name, config):
        document_name = f"{file_search_store_name}/documents/{config['display_name']}-{len(self.uploads)}"
        self.stores[file_search_store_name][document_name] = config["display_name"]
        self.uploads.append(config["display_name"])
        return SimpleNamespace(done=True, error=None, response=SimpleNamespace(document_name=document_name))

    def _delete_document(self, name, config=None):
        self.deleted_documents.append(name)
        for documents in self.stores.values():
            documents.pop(name, None)

    def _generate(self, model, contents, config):
        candidate = SimpleNamespace(grounding_metadata=self.grounding)
        return SimpleNamespace(text=self.answer_text, candidates=[candidate])


@pytest.fixture
def fake_client() -> FakeFileSearchClient:
    return FakeFileSearchClient()


def grounding(chunks: List[tuple], supports: List[tuple]):
    """Build grounding metadata: chunks=[(title, text)], supports=[(start, end, [chunk indices])]."""
    return SimpleNamespace(
        grounding_chunks=[
            SimpleNamespace(retrieved_context=SimpleNamespace(title=title, text=text)) for title, text in chunks
        ],
        grounding_supports=[
            SimpleNamespace(segment=SimpleNamespace(start_index=s, end_index=e), grounding_chunk_indices=idx)
            for s, e, idx in supports
        ],
    )
