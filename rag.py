#!/usr/bin/env python3
"""
RAG examples CLI — index documents and ask questions with any engine.

Engines:
  scratch            from scratch, markdown chunking (headings + size limit)
  scratch-semantic   from scratch, semantic chunking by gemini-3.5-flash-lite
  file-search        Gemini File Search (managed)

Usage:
  python rag.py index                              # scratch engine (default)
  python rag.py index --engine all                 # all three engines
  python rag.py ask "How many vacation days do I get?" --engine all
  python rag.py search "hotel limit in London"     # scratch retrieval only, no answer LLM call
  python rag.py ask "Montako lomapäivää saan?"     # translated into the base language first
  python rag.py search "hotel limit" --engine scratch-semantic
  python rag.py chunks travel-and-expenses.md --engine scratch-semantic   # preview chunking
  python rag.py status
  python rag.py reset --engine file-search         # deletes the remote store too
  python rag.py index --docs path/to/folder        # your own .md / .txt files
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import translation  # noqa: E402
from rag_common import (  # noqa: E402
    DEFAULT_DOCS_DIR,
    Answer,
    RagEngine,
    api_errors,
    load_documents,
    load_environment,
)
from translation import Query, Translator  # noqa: E402

SCRATCH_ENGINES = ("scratch", "scratch-semantic")
ENGINE_CHOICES = (*SCRATCH_ENGINES, "file-search", "all")


def make_engines(choice: str) -> List[RagEngine]:
    # Imported lazily so `--help` works without numpy or the Gemini SDK installed.
    from file_search.engine import FileSearchEngine
    from scratch.engine import ScratchEngine, semantic_engine

    factories: Dict[str, Callable[[], RagEngine]] = {
        "scratch": ScratchEngine,
        "scratch-semantic": semantic_engine,
        "file-search": FileSearchEngine,
    }
    names = factories if choice == "all" else [choice]
    return [factories[name]() for name in names]


def make_translator() -> Translator:
    return translation.default_translator()


def translate(question: str) -> Query:
    """Translate once up front so every engine (and the printout) shares the same query."""
    query = make_translator()(question)
    if query.translated:
        print(f"Translated ({query.language} → {query.base_language}): {query.text}")
    return query


def print_answer(answer: Answer, *, show_sources: bool) -> None:
    print(f"\n── {answer.engine} ({answer.seconds:.1f}s) " + "─" * 40)
    print(answer.text)
    if show_sources and answer.sources:
        print("\nSources:")
        for s in answer.sources:
            score = f" score={s.score:.3f}" if s.score is not None else ""
            mark = "*" if s.cited else " "
            print(f" {mark}[{s.number}] {s.document} — {s.title}{score}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG examples: from scratch (two chunking strategies) vs Gemini File Search")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_engine(p: argparse.ArgumentParser, default: str = "scratch") -> None:
        p.add_argument("--engine", choices=ENGINE_CHOICES, default=default)

    p_index = sub.add_parser("index", help="Index (or re-sync) a folder of documents")
    add_engine(p_index)
    p_index.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR, help="Folder of .md/.txt files")

    p_ask = sub.add_parser("ask", help="Answer a question from the indexed documents")
    add_engine(p_ask)
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--no-sources", action="store_true", help="Hide the source list")

    p_search = sub.add_parser("search", help="Scratch engine: show retrieved chunks and scores, no LLM")
    p_search.add_argument("question")
    p_search.add_argument("--engine", choices=SCRATCH_ENGINES, default="scratch")
    p_search.add_argument("--top-k", type=int, default=5)

    p_chunks = sub.add_parser("chunks", help="Preview how a scratch engine chunks one document (no indexing)")
    p_chunks.add_argument("file", type=Path, help="Document path, or a file name in --docs")
    p_chunks.add_argument("--engine", choices=SCRATCH_ENGINES, default="scratch")
    p_chunks.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR)

    add_engine(sub.add_parser("status", help="Show what is indexed"), default="all")
    p_reset = sub.add_parser("reset", help="Delete an index (File Search: deletes the remote store)")
    p_reset.add_argument("--engine", choices=ENGINE_CHOICES, required=True)

    args = parser.parse_args(argv)
    load_environment()

    try:
        if args.command == "index":
            documents = load_documents(args.docs)
            print(f"Indexing {len(documents)} document(s) from {args.docs}")
            for engine in make_engines(args.engine):
                print(engine.index(documents).summary())

        elif args.command == "ask":
            query = translate(args.question)
            for engine in make_engines(args.engine):
                print_answer(engine.ask(query, top_k=args.top_k), show_sources=not args.no_sources)

        elif args.command == "search":
            sources = make_engines(args.engine)[0].retrieve(translate(args.question), top_k=args.top_k)
            if not sources:
                print("No chunk scored above the similarity threshold.")
            for s in sources:
                print(f"\n[{s.number}] score={s.score:.3f}  {s.document} — {s.title}")
                print(textwrap.indent(textwrap.shorten(s.text, 300), "    "))

        elif args.command == "chunks":
            path = args.file if args.file.is_file() else args.docs / args.file
            if not path.is_file():
                raise ValueError(f"Document not found: {args.file}")
            chunks = make_engines(args.engine)[0].chunker(path.name, path.read_text(encoding="utf-8"))
            for chunk in chunks:
                print(f"\n[{chunk.id}] {len(chunk.text)} chars — {chunk.title}")
                print(textwrap.indent(textwrap.shorten(chunk.text, 200), "    "))

        elif args.command == "status":
            for engine in make_engines(args.engine):
                print(engine.status())

        elif args.command == "reset":
            for engine in make_engines(args.engine):
                engine.reset()
                print(f"{engine.name}: reset")

    except api_errors() as exc:  # e.g. empty index, missing API key, 429 rate limit
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
