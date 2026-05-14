#!/usr/bin/env python3
"""
app.py  —  Deliverable 6
End-to-end demo of the Vocabulary Recommender System.

Two modes:
  --mode cli    Interactive command-line demo (default)
  --mode web    Minimal HTTP server (no framework — stdlib http.server only)

External dependencies: none beyond the project's own modules + stdlib

Usage:
    python app.py --mode cli  \
        --model_dir outputs/best_checkpoint \
        --vectorstore data/vectorstore

    python app.py --mode web --port 7860   # then open http://localhost:7860
"""

import json
import argparse
import sys
import textwrap
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional

from data_pipeline import VocabularyList, VOCAB_LEVELS#
#from rag_pipeline import RAGPipeline
from rag_pipeline_v2 import RAGPipeline


# ---------------------------------------------------------------------------
# Vocabulary growth simulation state
# ---------------------------------------------------------------------------

class StudentSession:
    def __init__(self, level: str = "intermediate", custom: Optional[List[str]] = None):
        self.vocab          = VocabularyList(level=level, custom_words=custom)
        self.read_articles: List[str]  = []
        self.learned_words: List[str]  = []

    def mark_read(self, title: str, new_words: List[str]) -> None:
        if title not in self.read_articles:
            self.read_articles.append(title)
        for w in new_words:
            w = w.lower()
            if w not in self.vocab.known_words:
                self.vocab.known_words.add(w)
                self.learned_words.append(w)

    def stats(self) -> str:
        return (
            f"Level: {self.vocab.level} | "
            f"Known words: {len(self.vocab.known_words):,} | "
            f"Articles read: {len(self.read_articles)} | "
            f"Words learned: {len(self.learned_words)}"
        )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_recommendation(rec: Dict, width: int = 80) -> str:
    if "error" in rec:
        return f"\n  ERROR: {rec['error']}\n"
    sep = "─" * width

    title    = rec.get("recommended_title", "?")
    summary  = rec.get("summary", "")
    why      = rec.get("why_good_next_read", "")
    diff     = rec.get("difficulty_rating", "?")
    conf     = rec.get("confidence_score", 0.0)
    vocab    = rec.get("new_vocabulary", [])

    lines = [
        f"\n{sep}",
        f"  📖  {title}",
        sep,
        "",
        "  SUMMARY",
        *["  " + l for l in textwrap.wrap(summary, width - 4)],
        "",
        "  WHY THIS ARTICLE",
        *["  " + l for l in textwrap.wrap(why, width - 4)],
        "",
        f"  Difficulty: {diff}/10    Confidence: {conf:.0%}",
        "",
    ]
    if vocab:
        lines.append("  NEW VOCABULARY")
        for entry in vocab[:6]:
            if isinstance(entry, dict):
                w = entry.get("word", "")
                d = entry.get("definition", "")
                e = entry.get("example", "")
                lines.append(f"  • {w}: {d}")
                if e:
                    lines.append(f"    ↳ \"{e}\"")
    lines.append(sep + "\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

DEMO_PROFILES = {
    "1": ("beginner",     "animals in the ocean"),
    "2": ("intermediate", "renewable energy solar"),
    "3": ("advanced",     "machine learning neural networks"),
}


def run_cli(rag: RAGPipeline) -> None:
    print("\n" + "═" * 70)
    print("  Vocabulary Recommender System — Interactive Demo")
    print("  CSUSM CS322  |  LLaMA 3.1 8B + RAG")
    print("═" * 70)
    print("\nDemo profiles:")
    for k, (level, _) in DEMO_PROFILES.items():
        print(f"  [{k}] {level.capitalize()}")
    print("  [c] Custom vocabulary level")

    choice = input("\nSelect profile or press Enter for intermediate: ").strip()
    if choice in DEMO_PROFILES:
        level, default_query = DEMO_PROFILES[choice]
    elif choice == "c":
        level = input("Enter level (beginner/intermediate/advanced): ").strip()
        default_query = ""
    else:
        level, default_query = "intermediate", "science and technology"

    session = StudentSession(level=level)
    print(f"\n{session.stats()}")

    last_rec: Dict = {}

    while True:
        print("\nOptions: [q] query  [r] mark last as read  [s] stats  [x] exit")
        cmd = input("→ ").strip().lower()

        if cmd == "x":
            print("Goodbye!")
            break

        elif cmd == "s":
            print(f"\n  {session.stats()}")

        elif cmd == "r":
            if not last_rec:
                print("  No recommendation yet.")
                continue
            title = last_rec.get("recommended_title", "")
            new_w = [v["word"] for v in last_rec.get("new_vocabulary", [])
                     if isinstance(v, dict)]
            session.mark_read(title, new_w)
            print(f"\n  ✓ Read '{title}'")
            print(f"    +{len(new_w)} words added to vocabulary")
            print(f"  {session.stats()}")

        elif cmd == "q":
            query = input("  Topic query: ").strip()
            if not query:
                query = default_query or "science"
            cov_min = 0.90
            cov_max = 0.97
            print(f"\n  Searching (coverage window {cov_min*100:.0f}–{cov_max*100:.0f}%)…")
            rec = rag.recommend(
                query=query,
                vocab_list=session.vocab,
                coverage_min=cov_min,
                coverage_max=cov_max,
            )
            last_rec = rec
            print(_fmt_recommendation(rec))
            retrieved = rec.get("_retrieved_articles", [])
            if retrieved:
                print("  Retrieved articles:")
                for i, r in enumerate(retrieved, 1):
                    cov = r.get("coverage_ratio")
                    win = "✓" if r.get("vocab_in_window") else "✗"
                    print(f"    {i}. [{win}] {r.get('title','?')[:55]}"
                          + (f"  (cov={cov:.1%})" if cov else ""))
        else:
            print("  Unknown command.")


# ---------------------------------------------------------------------------
# Minimal web server — stdlib http.server only
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vocab Recommender</title>
<style>
  body{{font-family:sans-serif;max-width:860px;margin:40px auto;padding:0 20px;
        background:#f8fafc;color:#1e293b}}
  h1{{color:#2563eb}} label{{font-weight:600}}
  input,select,textarea{{width:100%;padding:8px;margin:6px 0 14px;border:1px solid #cbd5e1;
    border-radius:6px;font-size:14px;box-sizing:border-box}}
  button{{background:#2563eb;color:#fff;padding:10px 24px;border:none;border-radius:6px;
    cursor:pointer;font-size:15px}} button:hover{{background:#1d4ed8}}
  #result{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:20px;
    margin-top:20px;white-space:pre-wrap;font-family:monospace;font-size:13px}}
  .stat{{font-size:13px;color:#64748b;margin-bottom:12px}}
  .err{{color:#dc2626}}
</style>
</head>
<body>
<h1>📚 Vocabulary Recommender</h1>
<p class="stat">{stats}</p>
<form method="POST" action="/recommend">
  <label>Vocabulary level</label>
  <select name="level">
    <option value="beginner" {b}>Beginner (~1 000 words)</option>
    <option value="intermediate" {m}>Intermediate (~3 000 words)</option>
    <option value="advanced" {a}>Advanced (~6 000 words)</option>
  </select>
  <label>Topic query</label>
  <input name="query" value="{query}" placeholder="e.g. solar energy, ancient Rome…">
  <label>Coverage window</label>
  <input name="cov_min" value="{cov_min}" style="width:48%;display:inline-block">
  <input name="cov_max" value="{cov_max}" style="width:48%;display:inline-block;margin-left:4%">
  <button type="submit">Get Recommendation 🚀</button>
</form>
{mark_btn}
<div id="result">{result}</div>
</body></html>"""


def _make_handler(rag: RAGPipeline):
    session = StudentSession()
    last_rec: Dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass   # suppress access log

        def _send_html(self, html: str, code: int = 200) -> None:
            body = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _render(self, result: str = "", query: str = "",
                    cov_min: str = "90", cov_max: str = "97",
                    level: str = "intermediate") -> str:
            return _HTML_TEMPLATE.format(
                stats=session.stats(),
                b="selected" if level == "beginner" else "",
                m="selected" if level == "intermediate" else "",
                a="selected" if level == "advanced" else "",
                query=query,
                cov_min=cov_min, cov_max=cov_max,
                result=result,
                mark_btn=(
                    '<form method="POST" action="/mark_read" style="margin-top:10px">'
                    '<button type="submit" style="background:#16a34a">✅ Mark as Read</button>'
                    '</form>'
                ) if last_rec else "",
            )

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/"):
                self._send_html(self._render())
            else:
                self._send_html("<h3>Not found</h3>", 404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length).decode("utf-8")
            params = {k: v[0] for k, v in parse_qs(body).items()}
            path   = urlparse(self.path).path

            if path == "/mark_read":
                if last_rec:
                    title = last_rec.get("recommended_title", "")
                    wds   = [v["word"] for v in last_rec.get("new_vocabulary", [])
                             if isinstance(v, dict)]
                    session.mark_read(title, wds)
                    result = f"✓ Marked '{title}' as read. +{len(wds)} new words."
                else:
                    result = "No recommendation to mark."
                self._send_html(self._render(result))

            elif path == "/recommend":
                query   = params.get("query", "science").strip()
                level   = params.get("level", "intermediate")
                cov_min = params.get("cov_min", "90")
                cov_max = params.get("cov_max", "97")

                # Update session level if changed
                if session.vocab.level != level:
                    session.vocab = VocabularyList(level=level)

                try:
                    rec = rag.recommend(
                        query=query,
                        vocab_list=session.vocab,
                        coverage_min=float(cov_min) / 100,
                        coverage_max=float(cov_max) / 100,
                    )
                    last_rec.clear(); last_rec.update(rec)
                    result = _fmt_recommendation(rec).strip()
                except Exception as e:
                    result = f"ERROR: {e}"

                self._send_html(self._render(
                    result=result, query=query,
                    cov_min=cov_min, cov_max=cov_max, level=level,
                ))
            else:
                self._send_html("<h3>Not found</h3>", 404)

    return Handler


def run_web(rag: RAGPipeline, port: int = 7860) -> None:
    handler = _make_handler(rag)
    server  = HTTPServer(("0.0.0.0", port), handler)
    print(f"\n  Web demo running at  http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",         type=str, default="cli",
                   choices=["cli", "web"])
    p.add_argument("--model_dir",    type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--vectorstore",  type=str, default="data/vectorstore")
    p.add_argument("--port",         type=int, default=7860)
    p.add_argument("--gpu0_gb",      type=int, default=20)
    p.add_argument("--gpu1_gb",      type=int, default=22)
    args = p.parse_args()

    print("Loading RAG pipeline…")
    rag = RAGPipeline(
        vectorstore_dir=args.vectorstore,
        model_dir=args.model_dir,
        gpu0_gb=args.gpu0_gb,
        gpu1_gb=args.gpu1_gb,
    )

    if args.mode == "web":
        run_web(rag, port=args.port)
    else:
        run_cli(rag)

