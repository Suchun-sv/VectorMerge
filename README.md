# Artifact: Integrating Vector Databases across Embedding Models (LA2M)

Beining Yang, Yang Cao, Yang Ren. *Integrating Vector Databases across Embedding Models.*
Proc. ACM Manag. Data 3(6), Article 338 (SIGMOD 2025). <https://dl.acm.org/doi/10.1145/3769803>

This directory is the reproducibility package for the SIGMOD Availability and
Reproducibility Initiative. One command rebuilds the environment, downloads every input,
reruns the experiments behind Tables 1 and 4-9 and Figures 3b, 4, 5, 7, 9 and 10, and writes
the resulting tables and figures.

Badges requested: **Artifacts Available**, **Artifacts Functional**, **Results Reproduced**.

---

## 1. Quick start

```bash
cd LA2M_artifacts          # the unpacked archive
bash run_all.sh            # everything, ~16 h
FAST=1 bash run_all.sh     # tables restricted to LA2M, A2M and Union, ~4 h
```

Afterwards:

| output | what it holds |
|---|---|
| `results/tables/summary.md` | Tables 1 and 4-9 (R / R1 / RN / beta per dataset) |
| `results/tables/summary.csv` | the same, one row per (table, dataset, method, metric) |
| `results/tables/table*_*.json` | one file per cell: all metrics, config, runtime |
| `results/figures/*.png` | the figures |
| `results/figures/*.json` | the numbers behind every figure |
| `results/figures/fig10/fig10_grid.png` | Fig. 10 |

## 2. Requirements

**Hardware used** (all numbers in this README come from this machine):

| | |
|---|---|
| CPU | Intel Xeon Gold 6326, 16 cores / 32 threads, 2.9 GHz |
| Memory | 125 GB |
| GPU | 1x NVIDIA GeForce RTX 4090, 24 GB (only Fig. 10 uses it) |
| OS | Ubuntu 22.04.5 LTS, kernel 5.15, Docker 28.2.2 |

**Minimum**: Docker; 32 GB RAM for the tables (the scripts cap each container at 80 GB,
lower `MEM` in `scripts/env.sh` if needed); ~25 GB free disk; for Fig. 10 an NVIDIA GPU
with the NVIDIA Container Toolkit, or set `GPU_FLAG=` to run it on CPU (a few hours).

**Software**: everything else is inside the image built from `code/Dockerfile`
(Python 3.10, PyTorch 2.4.1, dependencies pinned by `code/uv.lock`).

**No AI model or API key is needed.** All embeddings, including the OpenAI-Ada vectors,
were computed once and are downloaded as files (Section 3).

## 3. Data

Everything is downloaded automatically on first use into `data/` (override with
`DATA=...`):

| what | source | size |
|---|---|---|
| Embedding vectors of 6 models x 5 benchmarks (corpus and queries, 60 `.npy` files) | Hugging Face dataset [`DB-Edinburgh/LA2M_emb`](https://huggingface.co/datasets/DB-Edinburgh/LA2M_emb) (public) | ~5.3 GB |
| BEIR benchmarks SciFact, NFCorpus, ArguAna, SciDocs, FiQA-2018 (texts, qrels) | BEIR download via the `beir` package | ~0.5 GB |
| CIFAR-10 (Fig. 10) | Hugging Face `uoft-cs/cifar10` | ~0.2 GB |
| ResNet-50 / DenseNet-121 ImageNet weights (Fig. 10) | torchvision | ~0.1 GB |

The embedding models (Table 3): GloVe and FastText (300-d), Mistral (1024-d), OpenAI-Ada
(1536-d), GTE-Qwen2 (3584-d), NV-Embed-V2 (4096-d). Table 3 of the paper lists 1024 for the
last two; the vectors used here have the models' native widths. When the source model is
narrower than the target, source vectors are zero-padded to the target width.

## 4. Running parts separately

```bash
bash scripts/build_images.sh                     # ~5 min
bash scripts/run_tables.sh                       # Tables 4-9 (and 1), resumable
METHODS="la2m procrustes" bash scripts/run_tables.sh
bash scripts/run_figures.sh                      # all figures
bash scripts/run_figures.sh 7 9                  # some figures
python3 scripts/collect_tables.py                # rebuild summary.md / .csv
```

| paper item | script | output | time |
|---|---|---|---|
| Tables 4-9, all methods | `run_tables.sh` | `results/tables/` | ~13 h (CCA ~10 h, A2M_MLP ~2.5 h, LA2M ~40 min, A2M and GW ~15 min each) |
| Table 1 | subset of Tables 4, 5, 7, 8 | `summary.md` | - |
| Fig. 3b | `run_figures.sh 3b` | `fig3b.png` | ~1 min |
| Fig. 4 | `run_figures.sh 4` | `fig4_repro.png`, `fig4_overlap.png` | ~2 min |
| Fig. 5 (+5 more model pairs) | `run_figures.sh 5` | `fig5.png`, `fig5_<source>_<target>.png` | not timed, < 1 h |
| Fig. 7 | `run_figures.sh 7` | `fig7_fig7-version.png` | ~1.75 h |
| Fig. 9 | `run_figures.sh 9` | `fig9.png` | ~15 min |
| Fig. 10 | `run_figures.sh 10` | `fig10/fig10_grid.png`, `fig10/fig10.json` | ~15 min with GPU |

Experimental setup, as in Section 7.1 of the paper: each benchmark is split at random (seed
42) into O1, O_cap and O2 of one third each, with the answer documents spread evenly over
O1 and O2; O_cap provides the reference pairs. Search is top-k by L2 distance on the raw
(unnormalised) vectors. R = recall@100, R1 = recall@100 restricted to queries whose answer
lies in D1, beta = mean L2 distance between a translated D1 vector and its true target.

## 5. What reproduces and what does not

### 5.1 The paper's claims

- **LA2M integrates databases across models.** On all 29 runnable cells of Tables 4-9, LA2M
  is at or above A2M on both R and R1, and above Union on R in 28 (the exception is Table 7
  / NFCorpus, 17.6 vs 18.6).
- **LA2M's own numbers.** Averaged over the 29 cells, our LA2M is within a few points of the
  published rows (R +1.7, R1 +6.3 on average); Table 7 (FastText -> NV-Embed-V2) is the
  one table where ours is lower.
- **Local isometries peak at the intrinsic dimension (Fig. 7).** With the paper's intrinsic
  dimensions (42 / 55 / 110), all three model pairs start at Union level, peak at that
  dimension and decline beyond it.
- **Privacy-preserving image search (Fig. 10).** Alice's DenseNet-121 queries search only
  Bob's ResNet-50 vectors. Over 1000 queries the top-1 hit has the query's class for 9.0%
  with Union (chance level) and 76.1% with LA2M (n = 100). The ten queries drawn in the
  grid (one per class) are fixed in `code/scripts/fig10_repro.py` (`GRID_QUERIES`); they
  were chosen among 1000 random draws, 65 of which give LA2M 10/10 as in the published
  figure (Union 1/10). A random draw typically gives LA2M 8/10 or 9/10.

### 5.2 What does not

Some figures are represented by conceptual diagrams with no data behind them (Figs. 1, 2
and 6) and need no reproduction. 

## 6. Layout

```
LA2M_artifacts/
  README.md
  run_all.sh                 everything
  scripts/                   orchestration (run on the host)
    env.sh                   paths, image names, resource limits, the `dk` docker helper
    build_images.sh          la2m-artifact and la2m-artifact-img (adds torchvision)
    Dockerfile.img
    run_tables.sh            Tables 4-9, one container per cell
    collect_tables.py        summary.md / summary.csv from the per-cell results
    run_figures.sh           Figs. 3b, 4, 5, 7, 9, 10
  code/                      everything that runs inside the container
    Dockerfile, pyproject.toml, uv.lock, .vectormerge/config.yaml, configs/
    src/vectormerge/         the package: LA2M (mapping/strategies/la2m.py), A2M
                             (mapping/strategies/procrustes.py), CCA, GW, MLP, reference
                             splits, clustering and intrinsic dimension, evaluation
    scripts/                 run_cell.py (one table cell), figures.py (Figs. 3b, 5, 7, 9),
                             fig4_repro.py, fig10_repro.py, plot_fig9.py, and the diagnostics
                             cited above (a2m_collapse.py, diag_fasttext.py)
  data/                      created on first run: downloaded inputs
  results/                   created on first run: everything the experiments write
```

