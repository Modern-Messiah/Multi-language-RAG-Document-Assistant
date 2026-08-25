import hashlib
import json
import sys
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# `streamlit run frontend/streamlit_app.py` puts frontend/ on sys.path, not the
# repository root, so the shared client package is not importable without this.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clients.backend import (  # noqa: E402  (must follow the sys.path fix)
    SUPPORTED_LANGUAGES,
    api_headers,
    backend_url,
    error_from_response,
    max_file_mb,
)

# Read .env like the bot and the backend do. Without this, a local
# `streamlit run` sent an empty X-API-Key and every request came back 401
# even though the key was sitting in .env. In docker the values arrive
# through env_file, where load_dotenv is a harmless no-op.
load_dotenv()

API_URL = backend_url()

# Mirrors the backend's MAX_FILE_SIZE so the UI never advertises a limit the
# API will not honour (the two read the same .env).
MAX_FILE_MB = max_file_mb()

# Turns sent with a question. The backend caps this again with
# MAX_HISTORY_TURNS; this is about not shipping a transcript that grows
# without bound across a long session.
HISTORY_TURNS_SENT = 6

# Shared secret for the backend; empty means the backend runs with auth
# disabled (development mode) and the header is simply ignored.
HEADERS = api_headers()

backend_error = error_from_response

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# =========================
# Custom CSS (Desktop + Mobile)
# =========================
st.markdown("""
<style>
/* =====================
   BASE (desktop)
===================== */
html, body {
    font-size: 20px;
}

input, textarea, button {
    font-size: 20px !important;
}

/* The answer used to be rendered into a hand-rolled div with hardcoded
   near-black colours, which meant dark text on a dark panel for anyone using
   Streamlit's light theme. st.chat_message follows the viewer's theme, so the
   .answer-box and .source-box rules that styled it are gone. */

section[data-testid="stSidebar"] * {
    font-size: 20px;
}

/* Streamlit's uploader prints its own "Limit 200MB per file" caption, which
   contradicts the backend's MAX_FILE_SIZE. Hide it; the label above states
   the real limit. (This rule was previously nested inside the block above,
   which made it - and the sidebar font-size - invalid CSS.) */
section[data-testid="stFileUploader"] small {
    display: none !important;
}

[data-testid="stCaptionContainer"] p {
    font-size: 20px;
    line-height: 1.6;
    color: #cbd5f5;
}

/* =====================
   MOBILE ADAPTATION
===================== */
@media (max-width: 768px) {

    html, body {
        font-size: 16px;
    }

    input, textarea, button {
        font-size: 16px !important;
        width: 100%;
    }

    section[data-testid="stSidebar"] {
        width: 100% !important;
    }

    h1 {
        font-size: 24px !important;
    }

    h2, h3 {
        font-size: 20px !important;
    }

    button {
        width: 100%;
    }

}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("📄 RAG Assistant")
st.caption(
    "Upload your documents and ask questions based **only on their content**"
)
st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# Per-session identity
# =========================
# Each browser session gets its own document namespace - previously all
# visitors shared one "streamlit_user" corpus and could clear each other's
# documents. Note: documents outlive the session on the backend; a page
# refresh starts a fresh namespace.
if "user_id" not in st.session_state:
    st.session_state["user_id"] = f"web-{uuid.uuid4().hex}"
USER_ID = st.session_state["user_id"]

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("📎 Upload documents")

    # The uploader is keyed off session_state so Clear can reset the widget:
    # bumping the key discards held files, otherwise the rerun after Clear
    # would instantly re-upload everything still sitting in the uploader.
    st.session_state.setdefault("uploader_key", 0)

    uploaded_files = st.file_uploader(
        f"TXT or PDF files (max {MAX_FILE_MB} MB per file)",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    # Streamlit reruns this script on every interaction, so both outcomes are
    # remembered: without that, each click re-uploaded every file still sitting
    # in the widget, and a file the backend rejected was retried forever.
    indexed_files = st.session_state.setdefault("indexed_files", set())
    failed_files = st.session_state.setdefault("failed_files", {})

    if uploaded_files:
        for file in uploaded_files:
            payload = file.getvalue()
            # Key on the content, not (name, size): two different notes.txt of
            # the same length would otherwise collide and the second would be
            # silently skipped.
            file_key = (file.name, hashlib.sha256(payload).hexdigest()[:16])
            if file_key in indexed_files or file_key in failed_files:
                continue

            # Reject oversize files here rather than spending two minutes
            # uploading something the backend is going to refuse.
            if len(payload) > MAX_FILE_MB * 1024 * 1024:
                failed_files[file_key] = (
                    f"too large ({len(payload) / (1024 * 1024):.1f} MB); "
                    f"the limit is {MAX_FILE_MB} MB"
                )
                continue

            if not payload:
                failed_files[file_key] = "the file is empty"
                continue

            with st.spinner(f"Processing {file.name}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (file.name, payload, file.type)},
                        params={"user_id": USER_ID},
                        headers=HEADERS,
                        timeout=120
                    )
                except requests.RequestException as e:
                    failed_files[file_key] = f"backend unreachable ({e})"
                    continue

            if response.status_code == 200:
                indexed_files.add(file_key)
                if response.json().get("duplicate"):
                    st.info(f"{file.name} already indexed")
                else:
                    st.success(f"{file.name} indexed")
            else:
                failed_files[file_key] = backend_error(response)

    # Failures persist across reruns, so report them from state rather than
    # only in the run that produced them.
    for (name, _digest), reason in failed_files.items():
        st.error(f"{name}: {reason}")

    st.divider()

    # =========================
    # What the backend actually holds
    # =========================
    # Asked every rerun rather than tracked in session_state: the widget only
    # ever knew about uploads made in this browser session, so after a refresh
    # the sidebar confidently showed nothing while the documents were still
    # indexed and still answering questions.
    st.header("📚 Your documents")

    indexed_documents = []
    try:
        listing = requests.get(
            f"{API_URL}/documents",
            params={"user_id": USER_ID},
            headers=HEADERS,
            timeout=30,
        )
    except requests.RequestException as e:
        st.warning(f"Could not reach the backend: {e}")
    else:
        if listing.status_code == 200:
            indexed_documents = listing.json()["documents"]
        else:
            st.warning(backend_error(listing))

    if not indexed_documents:
        st.caption("Nothing indexed yet.")
    else:
        for doc in indexed_documents:
            detail = f"{doc['chunks']} chunk{'s' if doc['chunks'] != 1 else ''}"
            if doc.get("pages"):
                detail += f", {doc['pages']} page{'s' if doc['pages'] != 1 else ''}"

            row, action = st.columns([5, 1])
            row.markdown(f"**{doc['source']}**  \n{detail}")
            if action.button("🗑", key=f"del_{doc['file_hash']}", help="Delete this document"):
                try:
                    removal = requests.delete(
                        f"{API_URL}/documents/{doc['file_hash']}",
                        params={"user_id": USER_ID},
                        headers=HEADERS,
                        timeout=30,
                    )
                except requests.RequestException as e:
                    st.error(f"Backend unreachable: {e}")
                else:
                    if removal.status_code == 200:
                        # The uploader still holds the file; without clearing
                        # that record the next rerun would re-upload it.
                        st.session_state.pop("indexed_files", None)
                        st.session_state["uploader_key"] += 1
                        st.rerun()
                    else:
                        st.error(backend_error(removal))

    st.divider()

    st.header("⚙️ Settings")

    # Labels decorate the values; the values themselves come from the backend's
    # language table, so the picker cannot offer something the chain ignores.
    LANG_FLAGS = {
        "Auto": "🌐",
        "English": "🇬🇧",
        "Русский": "🇷🇺",
        "Қазақша": "🇰🇿",
        "Français": "🇫🇷",
        "Deutsch": "🇩🇪",
        "Español": "🇪🇸",
        "日本語": "🇯🇵",
        "中文": "🇨🇳",
    }
    LANG_OPTIONS = {
        f"{name} {LANG_FLAGS.get(name, '')}".strip(): name
        for name in SUPPORTED_LANGUAGES
    }

    language_label = st.radio(
        "Answer language",
        list(LANG_OPTIONS.keys()),
        index=0
    )
    language = LANG_OPTIONS[language_label]

    st.divider()

    if st.button("🗑️ Clear all documents", use_container_width=True):
        try:
            resp = requests.post(
                f"{API_URL}/clear",
                params={"user_id": USER_ID},
                headers=HEADERS,
                timeout=60
            )
            if resp.status_code == 200:
                st.session_state.pop("indexed_files", None)
                st.session_state.pop("failed_files", None)
                st.session_state.pop("transcript", None)
                st.session_state["uploader_key"] += 1
                st.success("Cleared!")
                st.rerun()
            else:
                st.error(backend_error(resp))
        except requests.RequestException as e:
            st.error(f"Backend unreachable: {e}")

    st.info(
        "📌 **Limits**\n\n"
        "- Any number of documents\n"
        f"- Up to **{MAX_FILE_MB} MB per file**\n"
        "- Supported formats: **TXT, PDF**"
    )

# =========================
# Main - Status
# =========================
# Straight from the backend, which is the only thing that knows. Session
# state described this browser tab's uploads, so a refresh reported zero while
# the documents were still there, and a failed upload was counted as indexed.
if indexed_documents:
    count = len(indexed_documents)
    st.success(f"📚 {count} document{'s' if count != 1 else ''} indexed")
else:
    st.warning("No documents indexed yet")

# =========================
# Conversation
# =========================
# Every answer used to vanish the moment the next question was asked, and each
# question was sent on its own - so "and the second one?" retrieved on those
# literal words. The transcript lives in session state and rides along with the
# request, which keeps the API stateless.
transcript = st.session_state.setdefault("transcript", [])


def render_turn(turn):
    """Replay one exchange as chat bubbles.

    st.chat_message renders in the viewer's Streamlit theme. The old answer box
    was a hand-rolled div with hardcoded near-black colours, so on a light theme
    it was dark text on a dark panel.
    """
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn.get("sources"):
            with st.expander(f"📚 {len(turn['sources'])} source(s)"):
                for src in turn["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["preview"])


for past_turn in transcript:
    render_turn(past_turn)

question = st.chat_input(
    "Ask about your documents"
    if indexed_documents
    else "Upload a document first",
    disabled=not indexed_documents,
)

if question and question.strip():
    with st.chat_message("user"):
        st.markdown(question)

    payload = {
        "question": question,
        "language": language,
        "user_id": USER_ID,
        # Only what the backend will actually use; sending more would just be
        # prompt tokens the API discards.
        "history": [
            {"question": t["question"], "answer": t["answer"]}
            for t in transcript[-HISTORY_TURNS_SENT:]
        ],
    }

    with st.chat_message("assistant"):
        # The response is opened before streaming so an ordinary failure is
        # still an ordinary error message; only what arrives after a 200 has
        # to be reported inside the stream.
        try:
            response = requests.post(
                f"{API_URL}/query/stream",
                json=payload,
                headers=HEADERS,
                stream=True,
                timeout=120,
            )
        except requests.RequestException as e:
            st.error(f"Backend unavailable: {e}")
            st.stop()

        if response.status_code != 200:
            st.error(backend_error(response))
            st.stop()

        # Sources arrive before the first token, but the expander is rendered
        # under the answer, so they are collected as the stream runs.
        collected = {"sources": [], "error": None}

        def tokens():
            with response:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: "):])
                    kind = event.get("type")
                    if kind == "token":
                        yield event["text"]
                    elif kind == "sources":
                        collected["sources"] = event.get("sources", [])
                    elif kind == "error":
                        collected["error"] = event.get("detail", "The answer was cut short.")

        answer = st.write_stream(tokens())

        if collected["error"]:
            st.warning(collected["error"])

        if collected["sources"]:
            with st.expander(f"📚 {len(collected['sources'])} source(s)"):
                for src in collected["sources"]:
                    st.markdown(f"**{src['source']}**")
                    st.caption(src["preview"])

    transcript.append(
        {
            "question": question,
            "answer": answer,
            "sources": collected["sources"],
        }
    )
