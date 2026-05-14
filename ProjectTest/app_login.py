#!/usr/bin/env python3

import argparse
import textwrap
import json
import os

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from typing import Dict, List, Optional

from data_pipeline import VocabularyList
from rag_pipeline_v2 import RAGPipeline


# ---------------------------------------------------------------------------
# User database file
# ---------------------------------------------------------------------------

USERS_FILE = "users.json"

if not os.path.exists(USERS_FILE):

    with open(USERS_FILE, "w") as f:

        json.dump({}, f)


# ---------------------------------------------------------------------------
# Vocabulary Session State
# ---------------------------------------------------------------------------

class StudentSession:

    def __init__(
        self,
        level: str = "intermediate",
        custom: Optional[List[str]] = None
    ):

        self.vocab = VocabularyList(
            level=level,
            custom_words=custom
        )

        self.read_articles: List[str] = []

        self.learned_words: List[str] = []

    def mark_read(
        self,
        title: str,
        new_words: List[str]
    ):

        if title not in self.read_articles:

            self.read_articles.append(title)

        added = []

        for w in new_words:

            w = w.lower()

            if w not in self.vocab.known_words:

                self.vocab.known_words.add(w)

                self.learned_words.append(w)

                added.append(w)

        return added

    def stats(self):

        return (
            f"Level: {self.vocab.level} | "
            f"Known words: {len(self.vocab.known_words):,} | "
            f"Articles read: {len(self.read_articles)} | "
            f"Words learned: {len(self.learned_words)}"
        )


# ---------------------------------------------------------------------------
# Recommendation Formatting
# ---------------------------------------------------------------------------

def _fmt_recommendation(
    rec: Dict,
    width: int = 80
):

    if "error" in rec:

        return f"\nERROR: {rec['error']}\n"

    title = rec.get(
        "recommended_title",
        "Unknown"
    )

    summary = rec.get(
        "summary",
        ""
    )

    why = rec.get(
        "why_good_next_read",
        ""
    )

    diff = rec.get(
        "difficulty_rating",
        "?"
    )

    conf = rec.get(
        "confidence_score",
        0.0
    )

    vocab = rec.get(
        "new_vocabulary",
        []
    )

    lines = []

    lines.append(f"📖 {title}")

    lines.append("")

    lines.append("SUMMARY")

    lines.extend(
        textwrap.wrap(summary, width)
    )

    lines.append("")

    lines.append("WHY THIS ARTICLE")

    lines.extend(
        textwrap.wrap(why, width)
    )

    lines.append("")

    lines.append(
        f"Difficulty: {diff}/10"
    )

    lines.append(
        f"Confidence: {conf:.0%}"
    )

    if vocab:

        lines.append("")

        lines.append("NEW VOCABULARY")

        for entry in vocab[:10]:

            if isinstance(entry, dict):

                w = entry.get("word", "")

                d = entry.get("definition", "")

                lines.append(
                    f"• {w}: {d}"
                )

    return "\n".join(lines)

_STYLE_CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:"Poppins",sans-serif;
}

body{
    display:flex;
    justify-content:center;
    align-items:center;
    min-height:100vh;
    background:linear-gradient(135deg,#7c3aed,#4f46e5);
    background-size:cover;
    background-position:center;
}

.wrapper{
    width:420px;
    background:rgba(255,255,255,0.1);
    border:2px solid rgba(255,255,255,.2);
    backdrop-filter:blur(20px);
    box-shadow:0 0 10px rgba(0,0,0,.2);
    color:#fff;
    border-radius:10px;
    padding:30px 40px;
}

.wrapper h1{
    font-size:36px;
    text-align:center;
}

.wrapper .input-box{
    position:relative;
    width:100%;
    height:50px;
    margin:30px 0;
}

.input-box input{
    width:100%;
    height:100%;
    background:transparent;
    border:none;
    outline:none;
    border:2px solid rgba(255,255,255,.2);
    border-radius:40px;
    font-size:16px;
    color:#fff;
    padding:20px 45px 20px 20px;
}

.input-box input::placeholder{
    color:#fff;
}

.input-box i{
    position:absolute;
    right:20px;
    top:50%;
    transform:translateY(-50%);
    font-size:20px;
}

.remember-forgot{
    display:flex;
    justify-content:space-between;
    font-size:14.5px;
    margin:-15px 0 15px;
}

.remember-forgot label input{
    accent-color:#fff;
    margin-right:3px;
}

.remember-forgot a{
    color:#fff;
    text-decoration:none;
}

.remember-forgot a:hover{
    text-decoration:underline;
}

.btn{
    width:100%;
    height:45px;
    background:#fff;
    border:none;
    outline:none;
    border-radius:40px;
    box-shadow:0 0 10px rgba(0,0,0,.1);
    cursor:pointer;
    font-size:16px;
    color:#333;
    font-weight:600;
}

.register-link{
    font-size:14.5px;
    text-align:center;
    margin:20px 0 15px;
}

.register-link p a{
    color:#fff;
    text-decoration:none;
    font-weight:600;
}

.register-link p a:hover{
    text-decoration:underline;
}
"""

# ---------------------------------------------------------------------------
# Login Page
# ---------------------------------------------------------------------------

_LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta http-equiv="X-UA-Compatible" content="IE=edge">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Login Form</title>

<link href='https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css' rel='stylesheet'>

<style>
""" + _STYLE_CSS_TEMPLATE + """
</style>

</head>

<body>

<div class="wrapper">

<form method="POST" action="/login">

<h1>Login</h1>

<div class="input-box">

<input
    type="text"
    name="username"
    placeholder="Username"
    required
>

<i class='bx bxs-user'></i>

</div>

<div class="input-box">

<input
    type="password"
    name="password"
    placeholder="Password"
    required
>

<i class='bx bxs-lock-alt'></i>

</div>

<div class="remember-forgot">

<label>
<input type="checkbox">
Remember me
</label>

<a href="#">
Forgot password?
</a>

</div>

<button type="submit" class="btn">
Login
</button>

<div class="register-link">

<p>
Don't have an account?
<a href="#">
Register
</a>
</p>

</div>

</form>

</div>

</body>

</html>
"""


# ---------------------------------------------------------------------------
# Dashboard Template
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

<div style="position:absolute;top:20px;right:20px">

<a
href="/logout"

style="
background:#111827;
color:white;
padding:10px 16px;
border-radius:8px;
text-decoration:none;
font-size:14px;
font-family:sans-serif;
"
>

Logout

</a>

</div>

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



# ---------------------------------------------------------------------------
# Web Handler
# ---------------------------------------------------------------------------

def _make_handler(
    rag: RAGPipeline
):

    session = StudentSession()

    last_rec: Dict = {}

    logged_in = False

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, *_):
            pass

        def _send_html(
            self,
            html: str,
            code: int = 200
        ):

            body = html.encode("utf-8")

            self.send_response(code)

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(body)

        def _render(
            self,
            result: str = "",
            query: str = "",
            level: str = "intermediate",
            cov_min: str = "0.90",
            cov_max: str = "0.97"
        ):

            return _HTML_TEMPLATE.format(
 
                stats=session.stats(),

                b="selected" if level == "beginner" else "",

                m="selected" if level == "intermediate" else "",

                a="selected" if level == "advanced" else "",

                query=query,

                cov_min=cov_min,

                cov_max=cov_max,

                result=result,

                mark_btn=(
                    '<form method="POST" action="/mark_read">'
                    '<button type="submit">✅ Mark as Read</button>'
                    '</form>'
                ) if last_rec else ""
            )

        def do_GET(self):

            nonlocal logged_in

            if self.path == "/logout":

                logged_in = False

                self._send_html(_LOGIN_TEMPLATE)

                return

            if not logged_in:

                self._send_html(_LOGIN_TEMPLATE)

                return

            self._send_html(self._render())

        def do_POST(self):

            nonlocal logged_in

            length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(length).decode("utf-8")

            params = {
                k: v[0]
                for k, v in parse_qs(body).items()
            }

            path = urlparse(
                self.path
            ).path

            # LOGIN

            if path == "/login":

                username = params.get(
                    "username",
                    ""
                ).strip()

                password = params.get(
                    "password",
                    ""
                ).strip()

                with open(
                    USERS_FILE,
                    "r"
                ) as f:

                    users = json.load(f)

                if username not in users:

                    users[username] = {

                        "password": password,

                        "known_words": [],

                        "articles_read": [],

                        "vocab_level": "intermediate"
                    }

                    with open(
                        USERS_FILE,
                        "w"
                    ) as f:

                        json.dump(
                            users,
                            f,
                            indent=4
                        )

                else:

                    if users[username]["password"] != password:

                        self._send_html(
                            "<h1>Wrong Password</h1>"
                        )

                        return

                logged_in = True

                self._send_html(
                    self._render(
                        result=f"Welcome {username}!"
                    )
                )

                return

            # RECOMMEND

            elif path == "/recommend":

                query = params.get(
                    "query",
                    "science"
                ).strip()

                level = params.get(
                    "level",
                    "intermediate"
                )

                cov_min = params.get(
                    "cov_min",
                    "0.90"
                )

                cov_max = params.get(
                    "cov_max",
                    "0.97"
                )

                if session.vocab.level != level:

                    session.vocab = VocabularyList(
                        level=level
                    )

                rec = rag.recommend(
                    query=query,
                    vocab_list=session.vocab
                )

                last_rec.clear()

                last_rec.update(rec)

                result = _fmt_recommendation(rec)

                self._send_html(
                    self._render(
                        result=result,
                        query=query,
                        level=level,
                        cov_min=cov_min,
                        cov_max=cov_max
                    )
                )

                return

            # MARK READ

            elif path == "/mark_read":

                if last_rec:

                    title = last_rec.get(
                        "recommended_title",
                        ""
                    )

                    words = [

                        v["word"]

                        for v in last_rec.get(
                            "new_vocabulary",
                            []
                        )

                        if isinstance(v, dict)
                    ]

                    added = session.mark_read(
                        title,
                        words
                    )

                    result = (
                        f"✓ Marked '{title}' as read.\n"
                        f"+{len(added)} new words learned."
                    )

                else:

                    result = "No recommendation yet."

                self._send_html(
                    self._render(result=result)
                )

                return

    return Handler


# ---------------------------------------------------------------------------
# Run Web Server
# ---------------------------------------------------------------------------

def run_web(
    rag: RAGPipeline,
    port: int = 7860
):

    handler = _make_handler(rag)

    server = HTTPServer(
        ("0.0.0.0", port),
        handler
    )

    print(
        f"Web demo running at http://localhost:{port}"
    )

    server.serve_forever()


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        default="web"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    parser.add_argument(
        "--vectorstore",
        type=str,
        default="data/vectorstore"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860
    )

    args = parser.parse_args()

    print("Loading RAG pipeline…")

    rag = RAGPipeline(
        vectorstore_dir=args.vectorstore,
        model_dir=args.model_dir
    )

    run_web(
        rag,
        port=args.port
    )
