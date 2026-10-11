# Reproduction environment for
#   "Integrating Vector Databases across Embedding Models" (PACMMOD 3(6), SIGMOD 2025)
#   https://dl.acm.org/doi/10.1145/3769803
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t la2m-repro .
# Run a single table cell:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/output:/app/output" la2m-repro \
#       evaluate single-run -d scifact -stm mistral_openai -mm la2m

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline

# fasttext / pytrec-eval / faiss build from source and need a compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer: resolved strictly from uv.lock so the image is reproducible
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# Project layer
COPY src/ ./src/
COPY configs/ ./configs/
COPY .vectormerge/ ./.vectormerge/
RUN uv sync --frozen

# data/ and output/ are expected as bind mounts
RUN mkdir -p /app/data /app/output

ENTRYPOINT ["vectormerge"]
CMD ["--help"]
