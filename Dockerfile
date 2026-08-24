# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so a code change does not reinstall them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the unprivileged user before copying the code, so COPY can set the
# ownership directly. A `chown -R` after COPY would duplicate the whole tree
# into an extra image layer.
RUN useradd --create-home appuser

COPY --chown=appuser:appuser . .

# Named volumes mounted on data/ inherit this ownership on first use
RUN mkdir -p data/uploads data/chroma_db \
    && chown -R appuser:appuser data
USER appuser

# Expose ports (FastAPI=8000, Streamlit=8501)
EXPOSE 8000 8501

# No default command, it will be overridden in docker-compose
