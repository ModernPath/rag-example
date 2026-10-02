# AGENTS.md — RAG Examples Configuration

> This file defines the structure and rules for the RAG examples in this project.
> Read it before adding an engine, a chunking strategy, or a vector store.
> For concepts and a guided tour of the code, see [README.md](README.md).

---

## Folder Structure

```
rag-example/
├── rag.py                    # CLI for all engines (required entry point)
├── rag_common.py             # Config, Document loading, sync planning, result types, shared prompt rules
├── scratch/                  # From-scratch engine, one module per pipeline step
│   ├── chunking.py           # Step 1:  markdown chunking (pure functions, no LLM)
│   ├── semantic_chunking.py  # Step 1b: semantic chunking (LLM picks boundaries)
│   ├── embedder.py           # Step 2:  gemini-embedding-2
│   ├── vector_store.py       # Step 3:  ChromaDB, embedded
│   └── engine.py             # Step 4:  index / retrieve / ask
├── file_search/engine.py     # Gemini File Search engine (managed store)
├── ui/app.py                 # Flask side-by-side comparison (port 5020)
├── ui/templates/index.html
├── sample_docs/              # Fictional company documents (.md and .txt)
├── data/                     # Local indexes and store state (gitignored, keep .gitkeep)
├── tests/                    # Offline unit tests + opt-in live tests
├── .env.example              # documented environment variables (copy to .env)
├── requirements.txt
└── pytest.ini
```

---

## Registered Engines

| Engine | Built by | Chunking | Index location | Status |
|--------|----------|----------|----------------|--------|
| `scratch` | `ScratchEngine()` | `chunk_markdown` (headings + size limit) | `data/scratch/` (ChromaDB) | Active |
| `scratch-semantic` | `semantic_engine()` | `SemanticChunker` (`gemini-3.5-flash-lite`) | `data/scratch-semantic/` (ChromaDB) | Active |
| `file-search` | `FileSearchEngine()` | Google whitespace chunking | Remote store, state in `data/file_search/state.json` | Active |

Add a row here when you add an engine.

---

## Models

All model names come from `rag_common.py` and can be overridden with environment variables.
Never hard-code a model name in an engine.

| Purpose | Function | Variable | Default |
|---------|----------|----------|---------|
| Answer generation | `generation_model()` | `RAG_MODEL` | `gemini-3.8-flash` |
| Embeddings | `embedding_model()` | `RAG_EMBEDDING_MODEL` | `gemini-embedding-2` |
| Semantic chunking | `chunking_model()` | `RAG_CHUNKING_MODEL` | `gemini-3.5-flash-lite` |
| Data folder | `data_dir()` | `RAG_DATA_DIR` | `./data` |

API keys are read from `GOOGLE_AI_STUDIO_KEY`, `GEMINI_API_KEY`, or `GOOGLE_API_KEY`, loaded by
`load_environment()` from `.env` / `.env.local` in this folder or any parent (closest wins).
Document every new variable in `.env.example`. Never commit `.env` files (they are gitignored).

---

## Engine Contract

Every engine implements the `RagEngine` protocol from `rag_common.py`:

```python
class RagEngine(Protocol):
    name: str                                                    # CLI/UI identifier, kebab-case
    def index(self, documents: Sequence[Document]) -> IndexReport: ...
    def ask(self, question: str, *, top_k: int = 5) -> Answer: ...
    def status(self) -> Dict[str, object]: ...                   # must not create files or remote resources
    def reset(self) -> None: ...
```

An engine MUST:
- **Sync incrementally.** Use `plan_sync()` with content hashes: index new and changed documents,
  remove deleted ones, and skip unchanged ones (no embedding or LLM calls for them).
- **Answer in the shared style.** Use `ANSWER_RULES` as the system prompt and return
  `NOT_FOUND_REPLY` without calling the LLM when nothing relevant was retrieved.
- **Return numbered sources.** `Answer.sources` is a list of `Source(number, document, title, text,
  score, cited)`; `[n]` markers in the answer refer to `Source.number`, and `cited` is set for
  every source the answer references.
- **Raise `ValueError` or `RuntimeError` for user-facing errors** (empty index, missing key).
  The CLI and UI catch `api_errors()` and show the message instead of a traceback.
- **Take its collaborators as constructor arguments** (embedder, generator, chunker, client) so
  tests can pass fakes, with real Gemini implementations as defaults.

---

## Adding a Chunking Strategy

A chunker is any callable `(document_name, text) -> List[Chunk]`:

1. Create `scratch/{strategy}_chunking.py` with a numbered step docstring explaining *why* the
   strategy exists and how it works (match `chunking.py` and `semantic_chunking.py`).
2. Keep the cutting logic a **pure function** that is testable without an API key. Isolate any
   LLM call behind an injectable parameter (like `SemanticChunker(segment=...)`).
3. Chunk text MUST be copied verbatim from the document. If an LLM is involved, let it choose
   boundaries and titles only, never rewrite content.
4. Chunk ids are `"{document}#{n}"`, numbered from 0 per document.
5. Give the chunker a `name` attribute (shown in `status()` as `chunking`).
6. Register an engine for it in `scratch/engine.py`, following `semantic_engine()`, with its own
   `name` so it gets its own index folder. Then add it to `make_engines()` and `SCRATCH_ENGINES`
   in `rag.py`, `default_engines()` in `ui/app.py`, and the table above.

Changing an existing chunker does not re-index unchanged documents. After editing chunking code,
run `python rag.py reset --engine {engine}` and index again.

---

## Adding a Vector Store

Implement the six methods of `ChromaVectorStore` with the same signatures:
`__len__`, `document_hashes`, `add_document`, `remove_document`, `search`, and `reset`.
`search` returns `(Chunk, similarity)` pairs, best first, where higher means more similar.
The store MUST start fresh when the embedding model or dimensions differ from those it was
built with. `scratch/engine.py` should not need changes apart from `_open_store()`.

---

## Gemini Integration Rules

These rules come from bugs found while building this project (details in README *Gotchas*):

- **Keep the `genai.Client` referenced** while using it; the SDK closes its connection when the
  client is garbage-collected. Assign `client = get_client()` before calling it.
- **Disable automatic function calling** in every `generate_content` call without local tools:
  `automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`.
- **Batch embeddings with one `types.Content` per text.** `gemini-embedding-2` merges a plain
  list of strings into a single embedding.
- **Put the embedding task in the text**, not in `task_type` (ignored by `gemini-embedding-2`):
  use `format_query()` and `format_document()` from `scratch/embedder.py`.
- **Use structured output for machine-read responses** (`response_mime_type="application/json"`
  with `response_json_schema`), and validate the result before using it.
- **Grounding offsets are UTF-8 byte offsets.** Work on `text.encode("utf-8")` when inserting
  citations from `grounding_supports`.
- Requires **google-genai >= 2.27**.

---

## CLI Rules

`rag.py` is the single CLI. New features get a subcommand or a flag there, not a new script.

- `--engine` accepts the registered engine names plus `all`.
- Debug commands that skip the LLM (`search`, `chunks`) are part of the teaching value: keep them
  working for every scratch engine.
- Exit codes: 0 on success, 1 on a user-facing error printed to stderr as `Error: ...`.

```bash
python rag.py index  [--engine scratch|scratch-semantic|file-search|all] [--docs FOLDER]
python rag.py ask    "question" [--engine ...] [--top-k 5] [--no-sources]
python rag.py search "question" [--engine scratch|scratch-semantic] [--top-k 5]
python rag.py chunks FILE [--engine scratch|scratch-semantic]
python rag.py status [--engine ...]
python rag.py reset  --engine ...
```

---

## UI Rules

- `create_app(engines)` takes the engines as a mapping, so tests can pass fakes.
- Engines run in parallel; one engine failing MUST NOT break the others' results.
- Every registered engine appears as a status card and a selectable checkbox.
- Port 5020 (override with `PORT`). Colors are CSS variables with a dark-mode variant, and the
  layout must work at phone width.

---

## Testing

```bash
pytest                                          # offline, must pass without an API key
RAG_LIVE_TESTS=1 pytest tests/test_live.py      # real Gemini API
```

- Every change needs offline tests using the fakes in `tests/conftest.py`: `FakeEmbedder`,
  `FakeGenerator`, `FakeFileSearchClient`, and the `make_docs` fixture. Offline tests MUST NOT
  call the network.
- The autouse `isolated_data_dir` fixture points `RAG_DATA_DIR` at a temp folder, so tests never
  touch `data/`.
- Live tests are opt-in, MUST clean up after themselves (reset indexes, delete File Search
  stores), and should assert on facts from `sample_docs/` (e.g. "230" for the London hotel cap).
- After live runs, verify no `rag-example-*` File Search stores are left in the account.
- UI changes: check the page in a real browser as well as with the Flask test client.

---

## Sample Documents

`sample_docs/` holds fictional content only (company "Polar Pixel Oy"). When adding documents:
- Mark them as fictional sample content.
- Include specific, checkable facts (numbers, limits, deadlines) so answers can be verified.
- Don't contradict facts in existing documents.
- Supported formats are `.md` and `.txt` (see `SUPPORTED_SUFFIXES`).

---

## Code Style

- Python 3.10+, `from __future__ import annotations`, type hints, dataclasses for data.
- Each pipeline module starts with a docstring that explains the step to a student: what, why,
  and the trade-offs.
- Import the Gemini SDK and ChromaDB lazily (inside functions, or engines inside `make_engines()`),
  so `rag.py --help` works before dependencies are installed.
- Comments explain *why*, not *what*. Keep modules small; prefer pure functions for logic.
- Update `README.md` (comparison table, CLI reference, gotchas, exercises) whenever behavior
  changes.
