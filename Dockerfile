# =========================================================================
# Builder: compiles anything without a wheel, then is thrown away.
# =========================================================================
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Only the builder needs a toolchain. Keeping it out of the runtime image is
# the whole point of splitting the build in two.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install from the fully-resolved lock, not requirements.txt: the latter pins
# only the 17 direct dependencies and lets ~130 transitive ones float, which is
# how an incompatible posthog release started breaking startup.
# --require-hashes makes pip refuse anything whose artifact does not match.
COPY requirements.lock .
RUN pip install --require-hashes -r requirements.lock


# =========================================================================
# Runtime: no compiler, no lock, just the venv and the app.
# =========================================================================
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv

# libgomp1 is the one native library the runtime still needs: onnxruntime
# (pulled in by chromadb) links against libgomp and previously got it as a
# side effect of build-essential. ~150 KB instead of ~200 MB.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Create the unprivileged user before copying so COPY can set ownership
# directly; a `chown -R` afterwards would duplicate the tree in a new layer.
RUN useradd --create-home appuser

COPY --chown=appuser:appuser . .

# Named volumes mounted on data/ inherit this ownership on first use
RUN mkdir -p data/uploads data/chroma_db \
    && chown -R appuser:appuser data
USER appuser

# Expose ports (FastAPI=8000, Streamlit=8501)
EXPOSE 8000 8501

# No default command, it will be overridden in docker-compose
