"""RAG with Gemini File Search: Google runs chunking, embedding, storage, and retrieval.

    index:  documents ─► upload_to_file_search_store ─► (Google chunks + embeds)
    ask:    question ─► generate_content(tools=[FileSearch]) ─► answer + grounding metadata

Compare with scratch/engine.py: the pipeline is the same, but every step except
the final prompt is managed for you. What you still own here:
  - which documents are in the store (we sync by content hash, like the scratch engine)
  - chunking parameters (white_space_config below)
  - turning grounding metadata into citations

Local state (data/file_search/state.json) remembers the store name and which
document version was uploaded, so re-running `index` only uploads changes.
"""

from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import translation
from rag_common import (
    ANSWER_RULES,
    Answer,
    Document,
    IndexReport,
    Source,
    data_dir,
    embedding_model,
    generation_model,
    get_client,
    plan_sync,
)
from translation import Query, Translator, base_language, reply_language_rule

STORE_DISPLAY_NAME = "rag-example-sample-docs"
MAX_TOKENS_PER_CHUNK = 300
MAX_OVERLAP_TOKENS = 40
UPLOAD_TIMEOUT_SECONDS = 300
POLL_SECONDS = 2


class FileSearchEngine:
    name = "file-search"

    def __init__(
        self, directory: Optional[Path] = None, *, client: Any = None, translator: Optional[Translator] = None
    ) -> None:
        self.directory = Path(directory) if directory else data_dir() / "file_search"
        self._client = client
        self.translate = translator or translation.default_translator()

    @property
    def client(self) -> Any:
        # Created lazily and kept on the instance: the SDK closes its HTTP
        # connection when the client object is garbage-collected.
        if self._client is None:
            self._client = get_client()
        return self._client

    # -- indexing ----------------------------------------------------------- #

    def index(self, documents: Sequence[Document]) -> IndexReport:
        started = time.perf_counter()
        state = self._load_state()
        store_name = self._ensure_store(state)
        uploaded: Dict[str, Dict[str, str]] = state["documents"]
        plan = plan_sync(documents, {name: info["sha256"] for name, info in uploaded.items()})

        for name in [*plan.removed, *(doc.name for doc in plan.changed)]:
            self._delete_document(uploaded.pop(name)["document_name"])
            self._save_state(state)

        for doc in plan.to_index:
            uploaded[doc.name] = {"sha256": doc.sha256, "document_name": self._upload(store_name, doc)}
            self._save_state(state)  # save after each upload so an interrupted run can resume

        return IndexReport(
            engine=self.name,
            added=[d.name for d in plan.new],
            updated=[d.name for d in plan.changed],
            removed=plan.removed,
            unchanged=[d.name for d in plan.unchanged],
            seconds=time.perf_counter() - started,
        )

    def _ensure_store(self, state: Dict[str, Any]) -> str:
        """Reuse the saved store, or create one (also if it was deleted in the console)."""
        if state.get("store_name") and self._get_store(state["store_name"]) is not None:
            return state["store_name"]

        store = self.client.file_search_stores.create(
            config={
                "display_name": STORE_DISPLAY_NAME,
                "embedding_model": f"models/{embedding_model()}",
            }
        )
        state.update(store_name=store.name, documents={})
        self._save_state(state)
        return store.name

    def _upload(self, store_name: str, doc: Document) -> str:
        operation = self.client.file_search_stores.upload_to_file_search_store(
            file=str(doc.path),
            file_search_store_name=store_name,
            config={
                "display_name": doc.name,
                "mime_type": "text/plain",
                # Metadata can be used later to filter: metadata_filter='source="it-security.md"'
                "custom_metadata": [{"key": "source", "string_value": doc.name}],
                "chunking_config": {
                    "white_space_config": {
                        "max_tokens_per_chunk": MAX_TOKENS_PER_CHUNK,
                        "max_overlap_tokens": MAX_OVERLAP_TOKENS,
                    }
                },
            },
        )
        operation = self._wait(operation, label=doc.name)
        return operation.response.document_name

    def _wait(self, operation: Any, *, label: str) -> Any:
        deadline = time.monotonic() + UPLOAD_TIMEOUT_SECONDS
        while not operation.done:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Indexing {label} timed out after {UPLOAD_TIMEOUT_SECONDS}s.")
            time.sleep(POLL_SECONDS)
            operation = self.client.operations.get(operation)
        if operation.error:
            raise RuntimeError(f"Indexing {label} failed: {operation.error}")
        return operation

    def _delete_document(self, document_name: str) -> None:
        try:
            self.client.file_search_stores.documents.delete(name=document_name, config={"force": True})
        except Exception as exc:  # noqa: BLE001
            if not _is_not_found(exc):
                raise

    # -- querying ----------------------------------------------------------- #

    def ask(self, question: Union[str, Query], *, top_k: int = 5) -> Answer:
        from google.genai import types

        store_name = self._load_state().get("store_name")
        if not store_name:
            raise ValueError("The File Search store is empty. Run indexing first.")

        started = time.perf_counter()
        query = question if isinstance(question, Query) else self.translate(question)
        response = self.client.models.generate_content(
            model=generation_model(),
            contents=query.text,  # retrieval happens server-side, so it also sees the base language
            config=types.GenerateContentConfig(
                system_instruction=f"{ANSWER_RULES}\n{reply_language_rule(query)}",
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(file_search_store_names=[store_name], top_k=top_k)
                    )
                ],
                # File Search runs server-side; there are no Python functions to call locally.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        metadata = response.candidates[0].grounding_metadata if response.candidates else None
        sources = sources_from_grounding(metadata)
        text = insert_citations(response.text or "", metadata)
        return Answer(self.name, query.original, text.strip(), sources, time.perf_counter() - started, query=query)

    # -- housekeeping ------------------------------------------------------- #

    def status(self) -> Dict[str, object]:
        state = self._load_state()
        store = self._get_store(state["store_name"]) if state.get("store_name") else None
        return {
            "engine": self.name,
            "indexed": bool(store and state["documents"]),
            "documents": sorted(state.get("documents", {})),
            "active_documents": int(getattr(store, "active_documents_count", 0) or 0),
            "pending_documents": int(getattr(store, "pending_documents_count", 0) or 0),
            "size_bytes": int(getattr(store, "size_bytes", 0) or 0),
            "base_language": base_language(),
            "store": state.get("store_name"),
        }

    def reset(self) -> None:
        """Delete the remote store (and its documents) and forget local state."""
        store_name = self._load_state().get("store_name")
        if store_name:
            try:
                self.client.file_search_stores.delete(name=store_name, config={"force": True})
            except Exception as exc:  # noqa: BLE001
                if not _is_not_found(exc):
                    raise
        shutil.rmtree(self.directory, ignore_errors=True)

    def _get_store(self, store_name: str) -> Any:
        try:
            return self.client.file_search_stores.get(name=store_name)
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return None
            raise

    @property
    def _state_path(self) -> Path:
        return self.directory / "state.json"

    def _load_state(self) -> Dict[str, Any]:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        return {"store_name": None, "documents": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Grounding metadata → sources and inline citations (pure functions)
# --------------------------------------------------------------------------- #


def sources_from_grounding(metadata: Any) -> List[Source]:
    """One Source per retrieved chunk, numbered from 1, flagged if any sentence cites it."""
    chunks = getattr(metadata, "grounding_chunks", None) or []
    cited = {i for support in _supports(metadata) for i in support.grounding_chunk_indices or []}
    sources = []
    for index, chunk in enumerate(chunks):
        context = chunk.retrieved_context
        if context is None:
            continue
        sources.append(
            Source(
                number=index + 1,
                document=context.title or "",
                title=context.title or "",
                text=(context.text or "").strip(),
                cited=index in cited,
            )
        )
    return sources


def insert_citations(text: str, metadata: Any) -> str:
    """Append [n] markers after each grounded segment, like the scratch engine's citations.

    Segment offsets are UTF-8 byte positions, so we edit the encoded bytes and insert
    from the end of the text backwards to keep earlier offsets valid.
    """
    by_end: Dict[int, set] = defaultdict(set)
    for support in _supports(metadata):
        if support.segment is not None and support.segment.end_index is not None:
            by_end[support.segment.end_index].update(i + 1 for i in support.grounding_chunk_indices or [])

    encoded = text.encode("utf-8")
    for end in sorted(by_end, reverse=True):
        numbers = ", ".join(str(n) for n in sorted(by_end[end]))
        encoded = encoded[:end] + f" [{numbers}]".encode("utf-8") + encoded[end:]
    return encoded.decode("utf-8")


def _supports(metadata: Any) -> list:
    return list(getattr(metadata, "grounding_supports", None) or [])


def _is_not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or "NOT_FOUND" in str(exc)
