# 🤖 Multi-language RAG Document Assistant

A production-ready, multi-language **Retrieval-Augmented Generation (RAG)** assistant designed for querying multiple documents with accurate source attribution.

## Key Features

- **Multi-document Support**: Efficiently handles PDF and TXT file uploads.
- **Multilingual Intelligence**: Supports questions and answers in English, Russian, Kazakh, French, German, Spanish, Chinese, and Japanese.
- **Semantic Search**: Powered by **ChromaDB** for fast and relevant document retrieval.
- **Accurate Attribution**: Every answer comes with citations and text previews from the source documents.
- **Multiple Interfaces**:
  - **FastAPI Backend**: Robust and scalable API.
  - **Streamlit Web UI**: Responsive and user-friendly interface.
  - **Telegram Bot**: Query your documents on the go.
- **Dockerized**: Easy deployment using Docker and Docker Compose.

## Architecture Overview

The system follows a modular architecture:
- **Frontend**: Streamlit application for web-based interaction.
- **Bot**: Python-based Telegram bot implementation.
- **Backend**: FastAPI server orchestrating the RAG pipeline.
- **RAG Core**: Custom document loading, chunking, and retrieval logic using LangChain-like patterns.
- **Storage**: ChromaDB for vector embeddings and local filesystem for raw uploads.

## Quick Start

1. **Clone & Set up Environment**:
   ```bash
   git clone <repo-url>
   cp .env.template .env # Fill in your OPENAI_API_KEY
   ```

2. **Run with Docker (Recommended)**:
   ```bash
   docker-compose up --build
   ```

3. **Access the services**:
   - **Frontend**: `http://localhost:8501`
   - **API Docs**: `http://localhost:8000/docs`

4. **Telegram Bot**:
   - Start the bot by sending `/start` to `@MultiLanguageRAGBot`
   - Use `/upload` to upload documents and `/query` to ask questions.


![Demonstration of work](app/docs/assets/front.gif)

![Demonstration of work](app/docs/assets/tg.gif)

For detailed installation and technical details, see [DOCUMENTATION.md](DOCUMENTATION.md).
