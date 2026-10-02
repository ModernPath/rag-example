# RAG Examples: From Scratch vs Gemini File Search

Retrieval-Augmented Generation (RAG) answers questions from **your** documents:
find the relevant passages first, then ask the model to answer using only those
passages, with citations.

This project implements the same RAG app from scratch and with a managed service, behind
one CLI and one UI. The scratch engine comes with two chunking strategies, so you get three
engines to compare: `scratch` (markdown chunking), `scratch-semantic` (LLM chunking), and
`file-search`.

| | `scratch/` | `file_search/` |
|---|---|---|
| Chunking | Ours: markdown-aware (`scratch`) or semantic by `gemini-3.5-flash-lite` (`scratch-semantic`) | Google: whitespace chunks (size configurable) |
| Embeddings | `gemini-embedding-2`, called by us | `gemini-embedding-2`, called by Google at upload |
| Vector store | ChromaDB (open source, embedded, on local disk) | Managed File Search store |
| Retrieval | Cosine similarity, top-k, score threshold | Server-side, top-k |
| Generation | `gemini-3.8-flash` with a prompt we build | `gemini-3.8-flash` with the `FileSearch` tool |
| Citations | `[n]` markers the model writes, from numbered passages | Grounding metadata → `[n]` markers we insert |
| You can see | Every chunk, score, and the exact prompt | Retrieved chunks and grounding only |

Build them all, compare the answers side by side, and you'll understand what a managed
RAG service does for you, and what it hides.

## Quick start

Requires Python 3.10+ and a Gemini API key ([get one in AI Studio](https://aistudio.google.com/apikey)).

```bash
git clone https://github.com/ModernPath/rag-example.git
cd rag-example
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # needs google-genai >= 2.27
cp .env.example .env                   # then set GEMINI_API_KEY in .env

python rag.py index --engine all       # index sample_docs/ with all three engines
python rag.py ask "How many vacation days do I get after four years?" --engine all
python ui/app.py                       # http://localhost:5020 — compare side by side
```

The sample corpus in `sample_docs/` is a fictional company's handbook, travel policy,
IT security policy, and product FAQ (tidy markdown), plus an office guide copied from an
intranet web page (plain text, no headings, menu and footer included). Each has specific
facts (hotel caps, SLA percentages, deadlines), so you can check whether an answer is correct.

## The pipeline

```
           INDEX (once, and again when documents change)
 documents ──► chunk ──► embed ──► store
                                     │
           ASK (every question)      ▼
 question ──► embed ──► retrieve top-k ──► prompt with numbered passages ──► LLM ──► answer + [n] citations
```

### From scratch: read the code in this order

1. **`scratch/chunking.py`** splits markdown at headings and remembers the heading path
   (`Travel and Expense Policy > Hotels`). It packs whole paragraphs into chunks of
   ≤1200 chars and repeats the last paragraph in the next chunk (overlap), so facts near a
   boundary keep their context.
2. **`scratch/embedder.py`** turns text into 768-dim vectors with `gemini-embedding-2`.
   Read the module docstring: the new model has three traps (see *Gotchas*).
3. **`scratch/vector_store.py`** stores the vectors in [ChromaDB](https://www.trychroma.com/)
   running **embedded** (`chromadb.PersistentClient`): no server, everything persists in
   `data/scratch/`. We pass our own embeddings (`embedding_function=None`) and use cosine
   space. Chroma returns cosine *distance*, so similarity = `1 - distance`. Each chunk's
   metadata (document, heading path, document hash) also tells the engine which document
   versions are already indexed.
4. **`scratch/engine.py`** ties it together: incremental indexing, `retrieve()` (no LLM),
   and `ask()`, which builds the grounded prompt and flags which sources were cited. The
   chunker is a parameter: `ScratchEngine()` uses markdown chunking, `semantic_engine()`
   the semantic chunker, each with its own index (`data/scratch/`, `data/scratch-semantic/`).

Inspect retrieval without the LLM. This is the most useful debugging tool in RAG:

```bash
python rag.py search "Can I use a USB stick?"
# [1] score=0.674  it-security.md — IT Security Policy > Devices
# [2] score=0.617  travel-and-expenses.md — Travel and Expense Policy > Taxis ...
```

### Semantic chunking: `scratch/semantic_chunking.py`

Markdown chunking needs headings. A web page copied as text has none: one long paragraph
covers office access, visitors, and meeting rooms, with a menu above and a footer below.
Size-based cuts then land mid-topic, and every chunk gets the same title (the file name).

The semantic chunker numbers the document's sentences and asks `gemini-3.5-flash-lite` for
JSON: where each chunk starts, a descriptive title, and whether it is boilerplate.

```json
{"chunks": [{"start": 1, "title": "Navigation", "skip": true},
            {"start": 4, "title": "Helsinki office guide - Location and access badge", "skip": false},
            {"start": 9, "title": "Helsinki office guide - Visitor policy", "skip": false}, ...]}
```

We cut the **original** text at those sentences. The model never rewrites content (no
paraphrased or invented facts), its output is tiny (fast, cheap), and the boundaries are
sanitized so nothing is lost whatever it returns. Boilerplate chunks are dropped.

```bash
python rag.py chunks helsinki-office-guide.txt                            # markdown chunker
python rag.py chunks helsinki-office-guide.txt --engine scratch-semantic  # semantic chunker
python rag.py search "Can I bring a visitor?" --engine scratch            # 0.737, titled "helsinki-office-guide.txt"
python rag.py search "Can I bring a visitor?" --engine scratch-semantic   # 0.824, "Helsinki office guide - Visitor policy"
```

On the tidy markdown docs both strategies cut at roughly the same places. The trade-off:
one LLM call per document at index time (~3 s per document, run in parallel) and boundaries
that can vary a little between runs.

### Gemini File Search: `file_search/engine.py`

```python
store = client.file_search_stores.create(config={"display_name": "...", "embedding_model": "models/gemini-embedding-2"})

operation = client.file_search_stores.upload_to_file_search_store(
    file="sample_docs/it-security.md",
    file_search_store_name=store.name,
    config={"display_name": "it-security.md",
            "custom_metadata": [{"key": "source", "string_value": "it-security.md"}],
            "chunking_config": {"white_space_config": {"max_tokens_per_chunk": 300, "max_overlap_tokens": 40}}},
)
while not operation.done:            # indexing is asynchronous
    time.sleep(2)
    operation = client.operations.get(operation)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Can I use a USB stick?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(file_search=types.FileSearch(file_search_store_names=[store.name], top_k=5))]),
)
response.candidates[0].grounding_metadata   # retrieved chunks + which sentence cites which chunk
```

What you still own with the managed service:
- **Keeping the store in sync.** `data/file_search/state.json` maps each file's content hash
  to its remote document, so `index` re-uploads only changed files and deletes removed ones.
- **Chunking parameters.** Chunk size is the single biggest quality lever in every engine.
- **Citations.** `insert_citations()` turns `grounding_supports` into `[n]` markers.

The Google docs also show the newer **Interactions API**. It is marked experimental in
the SDK, so this example uses `generate_content`. The equivalent call:

```python
interaction = client.interactions.create(
    model="gemini-3.8-flash", input="Can I use a USB stick?",
    tools=[{"type": "file_search", "file_search_store_names": [store.name]}],
)
```

## CLI reference

```bash
python rag.py index  [--engine scratch|scratch-semantic|file-search|all] [--docs FOLDER]
python rag.py ask    "question" [--engine ...] [--top-k 5] [--no-sources]
python rag.py search "question" [--engine scratch|scratch-semantic] [--top-k 5]   # chunks + scores, no LLM
python rag.py chunks FILE [--engine scratch|scratch-semantic]                      # preview chunking, no index
python rag.py status [--engine ...]
python rag.py reset  --engine ...                     # file-search: deletes the remote store
```

Use your own documents: `python rag.py index --engine all --docs ~/my-notes` (`.md` and
`.txt`). File Search also accepts PDF, DOCX, and code files; extending the scratch engine
to PDFs is a good exercise.

| Variable | Default | Purpose |
|---|---|---|
| `RAG_MODEL` | `gemini-3.8-flash` | Generation model |
| `RAG_EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model (scratch + new File Search stores) |
| `RAG_CHUNKING_MODEL` | `gemini-3.5-flash-lite` | Model that picks chunk boundaries for `scratch-semantic` |
| `RAG_DATA_DIR` | `./data` | Where indexes and store state live |

## Tests

```bash
pytest                                          # offline: fake embedder, LLM, and File Search client
RAG_LIVE_TESTS=1 pytest tests/test_live.py      # real API; creates and deletes a temporary store
```

## Gotchas (all found while building this)

1. **`gemini-embedding-2` aggregates lists.** `embed_content(contents=["a", "b"])` returns
   **one** embedding for both texts combined. To batch, wrap each text in its own
   `types.Content`. Older code written for `gemini-embedding-001` silently breaks here.
2. **`task_type` is ignored by `gemini-embedding-2`.** Put the task in the text instead:
   `task: search result | query: …` for questions, `title: … | text: …` for documents.
3. **Similarity thresholds are model-specific.** With `gemini-embedding-2` at 768 dims,
   relevant passages here score ~0.65–0.85 and unrelated questions ~0.5–0.57, so the
   scratch engine uses 0.6. Recalibrate with `rag.py search` for your corpus.
4. **Grounding offsets are UTF-8 bytes, not characters.** With text like "10:00–14:00"
   (an en dash is 3 bytes), inserting citations by string index puts them in the wrong
   place.
5. **google-genai 2.27+** is needed for `embedding_model` when creating a File Search store.
   SDK 2.x also logs an "automatic function calling" warning on every `generate_content`
   unless you set `automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)`.
6. **Keep the `genai.Client` referenced while using it.** The SDK closes its HTTP
   connection when the client is garbage-collected, so chained one-liners can fail
   with "client has been closed".
7. **Rate limits.** A `429 RESOURCE_EXHAUSTED` can still surface after the SDK's
   retries. The CLI and UI report it as an error instead of crashing.
8. **Changing the chunker doesn't re-index.** Sync compares document hashes, so after
   editing chunking code or parameters run `rag.py reset --engine scratch` (or
   `scratch-semantic`) and index again. Changing the *embedding model* is detected
   automatically.

## Exercises

1. **Tune chunking.** Re-index with `max_chars=400` and then `3000`. Which questions get
   better or worse? Why do small chunks lose context and big chunks dilute similarity?
   Then compare `scratch` and `scratch-semantic` on questions about the office guide.
2. **Contextual chunks.** Extend the semantic chunker's JSON with a one-sentence `context`
   per chunk ("This section of the travel policy covers…") and embed it with the chunk.
   Does retrieval improve for short chunks?
3. **Hybrid search.** Add a keyword score (e.g. BM25 or simple term overlap) to the cosine
   score in `ChromaVectorStore.search`. Try a query with an exact product name or number.
4. **Metadata filtering.** Pass `metadata_filter='source="it-security.md"'` in `FileSearch`,
   and add the equivalent `document=` filter to the scratch engine.
5. **Multi-hop questions.** Ask *"I'm flying 7 hours to London. Which class and what hotel
   budget?"* The answer needs two sections. Inspect `top_k` and the retrieved sources.
6. **Make it agent memory.** Wrap `ScratchEngine.retrieve()` as a function tool for a
   Gemini agent (automatic function calling with a Python callable), so the agent decides
   when to look things up and cites what it found.
7. **Swap the vector store.** Point Chroma at a server (`chromadb.HttpClient`, run with
   `chroma run --path ./chroma-data`), or implement `ChromaVectorStore`'s six methods on
   Qdrant, LanceDB, or pgvector, without changing `engine.py`.

## Layout

```
rag-example/
├── AGENTS.md               # rules for extending this project (engines, chunkers, stores, tests)
├── rag.py                  # CLI for all engines
├── rag_common.py           # config, Document loading, sync planning, Answer/Source types
├── scratch/                # chunking.py | semantic_chunking.py → embedder.py → vector_store.py → engine.py
├── file_search/engine.py   # store sync, FileSearch tool, grounding → citations
├── ui/app.py               # Flask side-by-side comparison (port 5020)
├── sample_docs/            # fictional company documents (markdown + one web page as text)
├── data/                   # ChromaDB files + File Search state (gitignored)
├── .env.example            # copy to .env and set GEMINI_API_KEY
├── requirements.txt
└── tests/                  # offline unit tests + opt-in live tests
```
