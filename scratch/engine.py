"""Step 4 — Retrieve and generate: the from-scratch RAG engine.

    index:  documents ─► chunks ─► embeddings ─► ChromaDB (embedded, on disk)
    ask:    question ─► query embedding ─► top-k chunks ─► grounded prompt ─► Gemini

Everything the model sees is built here, so you can print the prompt, tweak the
chunking, or swap the embedder and watch how answers change.

The chunker is pluggable. Two engines are built from this class, each with its own index:
  scratch           markdown chunking (headings + size limit, no LLM)    -> data/scratch/
  scratch-semantic  semantic chunking (gemini-3.5-flash-lite picks cuts)  -> data/scratch-semantic/
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from rag_common import (
    ANSWER_RULES,
    NOT_FOUND_REPLY,
    Answer,
    Document,
    IndexReport,
    Source,
    data_dir,
    generate_text,
    plan_sync,
)
from scratch.chunking import Chunk, chunk_markdown
from scratch.embedder import Embedder, GeminiEmbedder
from scratch.semantic_chunking import SemanticChunker
from scratch.vector_store import ChromaVectorStore

# Below this cosine similarity a chunk is treated as unrelated to the question.
# Model-specific: with gemini-embedding-2 (768 dims) on the sample docs, relevant
# passages score ~0.65–0.85 and unrelated questions ~0.5–0.57. Recalibrate if you
# change the embedding model or corpus (use `rag.py search` to see the scores).
DEFAULT_MIN_SCORE = 0.6

CHUNKING_WORKERS = 4

CITATION_RULES = (
    "Cite the passages you used inline as [1], [2], … using their numbers. "
    "Never cite a passage you did not use."
)

Generate = Callable[..., str]  # (prompt, *, system) -> answer text
Chunker = Callable[[str, str], List[Chunk]]  # (document name, text) -> chunks


class ScratchEngine:
    def __init__(
        self,
        directory: Optional[Path] = None,
        *,
        name: str = "scratch",
        chunker: Chunker = chunk_markdown,
        embedder: Optional[Embedder] = None,
        generate: Optional[Generate] = None,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> None:
        self.name = name
        self.chunker = chunker
        self.directory = Path(directory) if directory else data_dir() / name
        self.embedder = embedder or GeminiEmbedder()
        self.generate = generate or generate_text
        self.min_score = min_score

    # -- indexing ----------------------------------------------------------- #

    def index(self, documents: Sequence[Document]) -> IndexReport:
        started = time.perf_counter()
        store = self._open_store()
        plan = plan_sync(documents, store.document_hashes())

        for name in [*plan.removed, *(doc.name for doc in plan.changed)]:
            store.remove_document(name)

        # Chunk documents in parallel (the semantic chunker makes one LLM call per document),
        # embed every new chunk in as few API calls as possible, then store per document.
        with ThreadPoolExecutor(max_workers=CHUNKING_WORKERS) as pool:
            per_doc = list(zip(plan.to_index, pool.map(lambda d: self.chunker(d.name, d.text), plan.to_index)))
        chunks = [c for _, doc_chunks in per_doc for c in doc_chunks]
        if chunks:
            vectors = self.embedder.embed_documents([c.title for c in chunks], [c.text for c in chunks])
            offset = 0
            for doc, doc_chunks in per_doc:
                store.add_document(doc.sha256, doc_chunks, vectors[offset : offset + len(doc_chunks)])
                offset += len(doc_chunks)

        return IndexReport(
            engine=self.name,
            added=[d.name for d in plan.new],
            updated=[d.name for d in plan.changed],
            removed=plan.removed,
            unchanged=[d.name for d in plan.unchanged],
            seconds=time.perf_counter() - started,
        )

    # -- querying ----------------------------------------------------------- #

    def retrieve(self, question: str, *, top_k: int = 5) -> List[Source]:
        """Retrieval only, no LLM: useful for debugging what the model will see."""
        store = self._open_store()
        if not len(store):
            raise ValueError("The scratch index is empty. Run indexing first.")
        hits = store.search(self.embedder.embed_query(question), top_k=top_k, min_score=self.min_score)
        return [
            Source(number=i, document=c.document, title=c.title, text=c.text, score=round(score, 4))
            for i, (c, score) in enumerate(hits, start=1)
        ]

    def ask(self, question: str, *, top_k: int = 5) -> Answer:
        started = time.perf_counter()
        sources = self.retrieve(question, top_k=top_k)
        if sources:
            text = self.generate(build_prompt(question, sources), system=f"{ANSWER_RULES}\n{CITATION_RULES}")
            mark_cited(text, sources)
        else:
            text = NOT_FOUND_REPLY  # nothing relevant retrieved: skip the LLM call
        return Answer(self.name, question, text, sources, time.perf_counter() - started)

    # -- housekeeping ------------------------------------------------------- #

    def status(self) -> Dict[str, object]:
        store = self._open_store() if self.directory.exists() else None  # don't create files just to report
        return {
            "engine": self.name,
            "indexed": bool(store and len(store)),
            "documents": sorted(store.document_hashes()) if store else [],
            "chunks": len(store) if store else 0,
            "chunking": self._chunker_id(),
            "embedding_model": self._embedder_id(),
            "location": str(self.directory),
        }

    def reset(self) -> None:
        self._open_store().reset()

    def _open_store(self) -> ChromaVectorStore:
        # Opened per call (Chroma caches the client per path, so this is cheap). The store
        # starts fresh if it was built with another embedder: those vectors are incompatible.
        return ChromaVectorStore(
            self.directory, embedding_model=self._embedder_id(), dimensions=self.embedder.dimensions
        )

    def _embedder_id(self) -> str:
        return getattr(self.embedder, "model", type(self.embedder).__name__)

    def _chunker_id(self) -> str:
        # SemanticChunker has .name; plain functions like chunk_markdown -> "markdown".
        return getattr(self.chunker, "name", None) or self.chunker.__name__.replace("chunk_", "")


def semantic_engine(directory: Optional[Path] = None, **kwargs) -> ScratchEngine:
    """The same engine with LLM-chosen chunk boundaries, in its own index (data/scratch-semantic/)."""
    kwargs.setdefault("chunker", SemanticChunker())
    return ScratchEngine(directory, name="scratch-semantic", **kwargs)


def build_prompt(question: str, sources: Sequence[Source]) -> str:
    passages = "\n\n".join(f"[{s.number}] ({s.document} — {s.title})\n{s.text}" for s in sources)
    return f"Documents:\n\n{passages}\n\nQuestion: {question}"


def mark_cited(answer_text: str, sources: Sequence[Source]) -> None:
    """Flag the sources whose [n] marker appears in the answer (handles "[1, 3]" too)."""
    cited = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", answer_text) for n in re.findall(r"\d+", group)}
    for source in sources:
        source.cited = source.number in cited
