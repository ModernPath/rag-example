"""Step 2 — Embedding: turn text into vectors whose distance reflects meaning.

Notes for gemini-embedding-2 (they differ from gemini-embedding-001):
  - `task_type` is ignored. Queries and documents are made asymmetric with text
    prefixes instead: "task: search result | query: ..." vs "title: ... | text: ...".
  - Passing several strings in `contents` returns ONE aggregated embedding.
    To batch, wrap each text in its own types.Content.
  - Outputs are already L2-normalized; we normalize anyway so the store also
    works with older models.
"""

from __future__ import annotations

from typing import List, Protocol, Sequence

import numpy as np

from rag_common import embedding_model, get_client

DEFAULT_DIMENSIONS = 768  # 768 / 1536 / 3072 are the recommended sizes
BATCH_SIZE = 100


class Embedder(Protocol):
    """Anything that maps text to unit vectors. Tests use a fake implementation."""

    dimensions: int

    def embed_documents(self, titles: Sequence[str], texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def format_document(title: str, text: str) -> str:
    return f"title: {title} | text: {text}"


def format_query(text: str) -> str:
    return f"task: search result | query: {text}"


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class GeminiEmbedder:
    def __init__(self, model: str | None = None, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        self.model = model or embedding_model()
        self.dimensions = dimensions

    def embed_documents(self, titles: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        inputs = [format_document(t, x) for t, x in zip(titles, texts)]
        batches = [self._embed(inputs[i : i + BATCH_SIZE]) for i in range(0, len(inputs), BATCH_SIZE)]
        return np.vstack(batches) if batches else np.zeros((0, self.dimensions), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([format_query(text)])[0]

    def _embed(self, inputs: List[str]) -> np.ndarray:
        from google.genai import types

        client = get_client()
        result = client.models.embed_content(
            model=self.model,
            # One Content per text => one embedding per text (not one aggregated vector).
            contents=[types.Content(parts=[types.Part(text=text)]) for text in inputs],
            config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
        )
        vectors = np.array([e.values for e in result.embeddings], dtype=np.float32)
        if len(vectors) != len(inputs):
            raise RuntimeError(f"Expected {len(inputs)} embeddings, got {len(vectors)}.")
        return normalize(vectors)
