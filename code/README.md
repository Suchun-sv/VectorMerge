# vectormerge

The LA2M package (vector database integration across embedding models) and the experiment
entry points used by the artifact. See `../README.md` for how to run the experiments.

- `src/vectormerge/` the package: mapping strategies (LA2M, A2M / Procrustes, CCA,
  Gromov-Wasserstein, MLP), reference splits, clustering, evaluation, embedding download
- `scripts/` experiment entry points (`run_cell.py` for one table cell, `figures.py` for
  Figs. 3b, 5, 7, 9, `fig4_repro.py`, `fig10_repro.py`) and plotting
- `Dockerfile`, `pyproject.toml`, `uv.lock` the pinned environment
- `.vectormerge/config.yaml` default configuration read by the package
