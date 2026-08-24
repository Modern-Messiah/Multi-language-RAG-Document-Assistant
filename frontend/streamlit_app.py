import html
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

# Read .env like the bot and the backend do. Without this, a local
# `streamlit run` sent an empty X-API-Key and every request came back 401
# even though the key was sitting in .env. In docker the values arrive
# through env_file, where load_dotenv is a harmless no-op.
load_dotenv()

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Mirrors the backend's MAX_FILE_SIZE so the UI never advertises a limit the
# API will not honour (the two read the same .env).
MAX_FILE_MB = int(os.getenv("MAX_FILE_SIZE", 30 * 1024 * 1024)) // (1024 * 1024)

# Shared secret for the backend; empty means the backend runs with auth
# disabled (development mode) and the header is simply ignored.
HEADERS = {"X-API-Key": os.getenv("BACKEND_API_KEY", "")}


def backend_error(response) -> str:
    """Human-readable reason from a failed backend response.

    Never dump response.text into the UI: a 422 body is a nested pydantic
    error blob, and .json() raises outright on a proxy's HTML error page.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # pydantic validation errors
        detail = "; ".join(str(item.get("msg", item)) for item in detail)
    return str(detail) if detail else f"Backend error (HTTP {response.status_code})"

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

.answer-box {
    background-color: #0f172a;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #334155;
    line-height: 1.65;
}

.source-box {
    background-color: #020617;
    padding: 14px;
    border-radius: 8px;
    border-left: 4px solid #38bdf8;
    margin-bottom: 12px;
}

section[data-testid="stSidebar"] * {
    font-size: 20px;
    section[data-testid="stFileUploader"] small {
    opacity: 0.25 !important;
    font-size: 11px !important;
    pointer-events: none;
}

section[data-testid="stFileUploader"] small {
    display: none !important;
}

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

    .answer-box {
        padding: 16px;
        font-size: 16px;
    }

    .source-box {
        padding: 12px;
        font-size: 15px;
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
# Each browser session gets its own document namespace — previously all
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
            file_key = (file.name, file.size)
            if file_key in indexed_files or file_key in failed_files:
                continue

            with st.spinner(f"Processing {file.name}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": file},
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
    for (name, _size), reason in failed_files.items():
        st.error(f"{name}: {reason}")

    st.divider()

    st.header("⚙️ Settings")

    LANG_OPTIONS = {
        "Auto 🌐": "Auto",
        "English 🇬🇧": "English",
        "Русский 🇷🇺": "Русский",
        "Қазақша 🇰🇿": "Қазақша",
        "Français 🇫🇷": "Français",
        "Deutsch 🇩🇪": "Deutsch",
        "Español 🇪🇸": "Español",
        "日本語 🇯🇵": "日本語",
        "中文 🇨🇳": "中文",
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
# Main — Status
# =========================
# Count what the backend actually accepted, not what the widget is holding:
# the old count included files that failed to index, and dropped to zero as
# soon as a file was removed from the uploader even though it stayed indexed.
if indexed_files:
    st.success(f"📚 {len(indexed_files)} document(s) indexed")
else:
    st.warning("No documents uploaded yet")

# =========================
# Question input
# =========================
st.subheader("💬 Ask a question")

question = st.text_input(
    "Type your question",
    placeholder="What is RAG?"
)

ask_btn = st.button("🔍 Ask", type="primary")

# =========================
# Ask logic
# =========================
if ask_btn:
    # Gate on what is indexed, not on the uploader widget: removing a file
    # from the widget used to block questions about documents that are still
    # indexed on the backend.
    if not indexed_files:
        st.warning("Please upload at least one document first")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": question,
                    "language": language,
                    "user_id": USER_ID
                },
                headers=HEADERS,
                timeout=120
            )
        except requests.RequestException as e:
            st.error(f"Backend unavailable: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(backend_error(response))
    else:
        data = response.json()

        st.subheader("🧠 Answer")
        # Escape the answer before dropping it into the styled div: it is
        # model output grounded in user-uploaded documents, so an uploaded
        # file could otherwise inject markup into the page.
        answer_html = html.escape(data["answer"]).replace("\n", "<br>")
        st.markdown(
            f"<div class='answer-box'>{answer_html}</div>",
            unsafe_allow_html=True
        )

        if data.get("sources"):
            st.subheader("📚 Sources")
            st.caption("Sources used to generate the answer")

            for src in data["sources"]:
                # Use st.expander or safe rendering for sources to avoid InvalidCharacterError
                with st.expander(f"📍 {src['source']}"):
                    st.write(src['preview'])
