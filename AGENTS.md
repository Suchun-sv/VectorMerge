# AGENTS.md

## What this repo is (agent-critical)
- Python package `vectormerge` for BEIR dataset handling, embedding generation, reference split creation, clustering, vector-space mapping, and evaluation.
- CLI entrypoint is `vectormerge` from `pyproject.toml` (`vectormerge.cli:app`).

## Use these commands (verified)
- Environment/bootstrap (from `install.sh`):
  - `uv venv`
  - `uv sync`
  - `uv pip install -e .`
- Typical data + embedding prep:
  - `uv run vectormerge dataset download -d all`
  - `uv run vectormerge embedding generate -m all -d all`

## Real CLI surface (prefer code over README)
README command names are partially stale; use the Typer wiring in `src/vectormerge/cli/__init__.py`:
- `vectormerge embedding generate ...`
- `vectormerge create-reference ...` (top-level command, not `reference create`)
- `vectormerge create-cluster ...` (invokes callback directly)
- `vectormerge map-embedding map ...`
- `vectormerge evaluate single-run ...`
- Backward-compat commands still exist:
  - `vectormerge download-dataset ...`
  - `vectormerge list-datasets`
  - `vectormerge list-models`

## Config resolution and gotchas
- Config merge precedence in code: package -> global -> project (`./.vectormerge/config.yaml` wins).
- This repo contains `./.vectormerge/config.yaml`, so local defaults are active during development.
- Dynamic overrides are supported via unknown CLI args parsed as dotted keys, e.g.:
  - `--mapping_config.la2m_config.num_clusters=50`
- Important quirk: `ConfigLoader.package_config_path` points under `src/configs/config.yaml`; repo root `configs/config.yaml` is not the path used by that loader method.

## Data/artifact expectations
- DVC tracks `data/` (`data.dvc` ~6.5GB metadata). Large data is expected.
- `.dvc/config` points to repo-specific cache/remote paths (e.g. `/mnt/work/...`, `s3://dvc-local-cache/...`), which may not exist on another machine.
- Default artifact locations are under:
  - `data/raw/beir`
  - `data/processed/{embeddings,references,clusters,mappings}`
  - `output/{mapping_models,mapping_embeddings}`

## External credentials and model behavior
- `.env.example` defines `OPENAI_API_KEY` and `MISTRAL_API_KEY`.
- `embedding_generator.py` loads `.env` automatically (`load_dotenv()`).
- API-backed models (`openai`, `mistral`) require keys; some local models may trigger large first-time downloads (e.g. Word2Vec/GloVe).

## CI / verification reality
- No repo-configured test/lint/typecheck pipeline found (no pytest/ruff/mypy config files).
- Only GitHub workflow present is docs deployment (`.github/workflows/deploy-docs.yml`), building Sphinx and publishing Pages.
- If you change code, do focused CLI smoke checks instead of assuming a full automated test suite exists.

## Productization priorities (non-paper usage)
- Extensibility is first-class: do not hardcode BEIR-only assumptions in new features.
- Keep integrations behind registries/adapters; prefer add-by-registration over edit-many-files wiring.
- Target registries: dataset registry (loader/provider abstraction), model registry (embedding backend abstraction), method registry (mapping strategy abstraction).
- Python library integration is first-class: CLI is a shell layer, not the only runtime interface.
- Put orchestration and business logic in reusable Python APIs (`runner/core`-style modules), not directly inside CLI handlers.
- CLI commands should call stable library entrypoints and return structured results suitable for Airflow/services/notebooks, not only console text.
