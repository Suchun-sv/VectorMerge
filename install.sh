~/.local/bin/uv venv
~/.local/bin/uv sync
~/.local/bin/uv pip install -e .
~/.local/bin/uv run vectormerge embedding generate -m all -d all 