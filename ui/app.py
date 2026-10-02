#!/usr/bin/env python3
"""
Flask UI: ask the same question to the RAG engines and compare answers side by side.

Run:
  python ui/app.py            # http://localhost:5020
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from flask import Flask, flash, redirect, render_template, request, url_for

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_common import RagEngine, api_errors, load_documents, load_environment  # noqa: E402

DEFAULT_PORT = 5020
DEFAULT_TOP_K = 5
EXAMPLE_QUESTIONS = (
    "How many vacation days do I get after four years?",
    "I'm flying 7 hours and staying in London. Which class and what hotel budget?",
    "Which Northstar plan do I need for SSO, and what is its uptime SLA?",
    "What should I do if I lose my laptop?",
    "Can I bring a visitor to the Helsinki office?",
    "What is the capital of Peru?",
)


def default_engines() -> Dict[str, RagEngine]:
    from file_search.engine import FileSearchEngine
    from scratch.engine import ScratchEngine, semantic_engine

    engines: List[RagEngine] = [ScratchEngine(), semantic_engine(), FileSearchEngine()]
    return {engine.name: engine for engine in engines}


def create_app(engines: Optional[Mapping[str, RagEngine]] = None) -> Flask:
    """App factory: pass fake engines in tests."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "rag-example-dev-secret")
    engines = dict(engines or default_engines())
    errors = api_errors()

    def safe_status(engine: RagEngine) -> dict:
        try:
            return engine.status()
        except errors as exc:
            return {"engine": engine.name, "indexed": False, "error": str(exc)}

    def ask_one(engine: RagEngine, question: str, top_k: int) -> dict:
        try:
            return {"answer": engine.ask(question, top_k=top_k)}
        except errors as exc:
            return {"error": str(exc)}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/")
    def index():
        question = request.args.get("q", "").strip()
        selected = request.args.getlist("engine") or list(engines)
        top_k = request.args.get("top_k", DEFAULT_TOP_K, type=int)

        results: Dict[str, dict] = {}
        if question:
            chosen = [engines[name] for name in selected if name in engines]
            # Engines are independent network calls: run them in parallel.
            with ThreadPoolExecutor(max_workers=len(chosen) or 1) as pool:
                futures = {e.name: pool.submit(ask_one, e, question, top_k) for e in chosen}
            results = {name: future.result() for name, future in futures.items()}

        return render_template(
            "index.html",
            statuses={name: safe_status(engine) for name, engine in engines.items()},
            question=question,
            selected=selected,
            top_k=top_k,
            results=results,
            examples=EXAMPLE_QUESTIONS,
        )

    @app.post("/index/<name>")
    def run_index(name: str):
        engine = engines.get(name)
        if engine is None:
            flash(f"Unknown engine: {name}", "error")
        else:
            try:
                flash(engine.index(load_documents()).summary(), "success")
            except errors as exc:
                flash(f"{name}: {exc}", "error")
        return redirect(url_for("index"))

    return app


if __name__ == "__main__":
    load_environment()
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    create_app().run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
