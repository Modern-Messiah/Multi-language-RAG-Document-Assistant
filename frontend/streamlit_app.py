import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

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
# Sidebar
# =========================
with st.sidebar:
    st.header("📎 Upload documents")

    uploaded_files = st.file_uploader(
        "TXT or PDF files (max 30 MB per file)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": file}
                )

            if response.status_code == 200:
                st.success(f"{file.name} indexed")
            else:
                st.error(response.text)

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

    st.info(
        "📌 **Limits**\n\n"
        "- Any number of documents\n"
        "- Up to **30 MB per file**\n"
        "- Supported formats: **TXT, PDF**"
    )

# =========================
# Main — Status
# =========================
if uploaded_files:
    st.success(f"📚 {len(uploaded_files)} document(s) indexed")
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
    if not uploaded_files:
        st.warning("Please upload at least one document first")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Thinking..."):
        response = requests.post(
            f"{API_URL}/query",
            json={
                "question": question,
                "language": language
            }
        )

    if response.status_code != 200:
        st.error(response.text)
    else:
        data = response.json()

        st.subheader("🧠 Answer")
        st.markdown(
            f"<div class='answer-box'>{data['answer']}</div>",
            unsafe_allow_html=True
        )

        if data.get("sources"):
            st.subheader("📚 Sources")
            st.caption("Sources used to generate the answer")

            for src in data["sources"]:
                st.markdown(
                    f"""
                    <div class="source-box">
                        <b>{src['source']}</b><br>
                        <small>{src['preview']}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
