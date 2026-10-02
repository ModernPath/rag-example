"""Step 3 — Vector store: keep chunk vectors and find the nearest ones to a query.

Uses ChromaDB, an open-source vector database, in embedded mode: no server to run,
everything persists to a folder (data/scratch/chroma.sqlite3 + index files).

We pass our own Gemini embeddings in (embedding_function=None), so Chroma only
stores vectors and runs the nearest-neighbour search. Each chunk is stored with
metadata (document, heading path, document hash), which also tells the engine
which document versions are already indexed — no separate manifest file.

Same idea works with a Chroma server (chromadb.HttpClient), Qdrant, pgvector, …
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from scratch.chunking import Chunk

COLLECTION = "chunks"


class ChromaVectorStore:
    def __init__(self, directory: Path, *, embedding_model: str, dimensions: int) -> None:
        import chromadb

        self.directory = Path(directory)
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.client = chromadb.PersistentClient(
            path=str(self.directory), settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()

    def _open_collection(self):
        """Open the collection, recreating it if it was built with a different embedder."""
        expected = {"embedding_model": self.embedding_model, "dimensions": self.dimensions}
        existing = {c.name for c in self.client.list_collections()}
        if COLLECTION in existing:
            collection = self.client.get_collection(COLLECTION)
            if {k: (collection.metadata or {}).get(k) for k in expected} == expected:
                return collection
            # Vectors from another model/size are not comparable: start over.
            self.client.delete_collection(COLLECTION)
        return self.client.create_collection(
            COLLECTION,
            embedding_function=None,  # we supply embeddings ourselves
            configuration={"hnsw": {"space": "cosine"}},
            metadata=expected,
        )

    def __len__(self) -> int:
        return self.collection.count()

    def document_hashes(self) -> Dict[str, str]:
        """{document name: sha256} of every document currently in the store."""
        metadatas = self.collection.get(include=["metadatas"])["metadatas"] or []
        return {m["document"]: m["sha256"] for m in metadatas}

    def add_document(self, sha256: str, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must match.")
        if not chunks:
            return
        if vectors.shape[1] != self.dimensions:
            raise ValueError(f"Expected {self.dimensions}-dim vectors, got {vectors.shape[1]}.")
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[{"document": c.document, "title": c.title, "sha256": sha256} for c in chunks],
        )

    def remove_document(self, document: str) -> None:
        self.collection.delete(where={"document": document})

    def search(
        self, query_vector: np.ndarray, *, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[Chunk, float]]:
        """Return up to top_k (chunk, cosine similarity) pairs, best first."""
        count = len(self)
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            score = 1.0 - distance  # Chroma returns cosine *distance*
            if score >= min_score:
                hits.append((Chunk(chunk_id, meta["document"], meta["title"], text), score))
        return hits

    def reset(self) -> None:
        if COLLECTION in {c.name for c in self.client.list_collections()}:
            self.client.delete_collection(COLLECTION)
        self.collection = self._open_collection()
