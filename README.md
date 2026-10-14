# Artifact: Integrating Vector Databases across Embedding Models (LA2M)

Beining Yang, Yang Cao, Yang Ren. *Integrating Vector Databases across Embedding Models.*
Proc. ACM Manag. Data 3(6), Article 338 (SIGMOD 2025). <https://dl.acm.org/doi/10.1145/3769803>

This directory is the reproducibility package for the SIGMOD Availability and
Reproducibility Initiative. One command rebuilds the environment, downloads every input,
reruns the experiments behind Tables 1 and 4-9 and Figures 3b, 5, 7, 8, 9 and 10, and writes
the resulting tables and figures.

Badges requested: **Artifacts Available**, **Artifacts Functional**, **Results Reproduced**.

---

## 1. Quick start

```bash
cd LA2M_artifacts          # the unpacked archive
bash run_all.sh            # everything, ~6 h
```

Afterwards:

| output | what it holds |
|---|---|
| `results/report.pdf` | every reproduced table and figure in the paper's layout, numbering and captions, to read side by side with the paper |
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

**Minimum**: Docker; 96 GB RAM, since the largest cells (FiQA-2018 into the 4096-d
NV-Embed-V2 space) need more than 64 GB (the scripts cap each container at 80 GB via `MEM`
in `scripts/env.sh`); ~25 GB free disk; for Fig. 10 an NVIDIA GPU
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
(source scripts/env.sh; dk -- scripts/make_report.py)   # rebuild results/output/report.pdf
```

| paper item | script | output | time |
|---|---|---|---|
| Tables 4-9, all methods | `run_tables.sh` | `results/tables/` | ~4 h (A2M_MLP ~2.2 h, CCA ~1 h, LA2M ~25 min, A2M and GW ~10 min each) |
| Table 1 | subset of Tables 4, 5, 7, 8 | `summary.md` | - |
| Fig. 3b | `run_figures.sh 3b` | `fig3b.png` | ~10 s |
| Fig. 5 (+5 more model pairs) | `run_figures.sh 5` | `fig5.png`, `fig5_<source>_<target>.png` | ~2 min |
| Fig. 7 | `run_figures.sh 7` | `fig7_fig7-version.png` | ~1.75 h |
| Fig. 8 | `run_figures.sh 8` | `fig8.png`, `fig8/` | ~20 min |
| Fig. 9 | `run_figures.sh 9` | `fig9.png` | ~15 min |
| Fig. 10 | `run_figures.sh 10` | `fig10/fig10_grid.png`, `fig10/fig10.json` | ~1 min with GPU, plus the CIFAR-10 download on first run |

Experimental setup, as in Section 7.1 of the paper: each benchmark is split at random (seed
42) into O1, O_cap and O2 of one third each, and O_cap provides the reference pairs. The
source database D1 holds O1 and O_cap, the target database D2 holds O2, so the integrated
database is the translated source vectors of O1 and O_cap plus the target vectors of O2
(for Union, the source vectors untranslated). The answer documents fall wherever the random
split puts them, about a third in each part. Search is top-k by L2 distance on the raw
(unnormalised) vectors. R = recall@100, R1 = recall@100 restricted to queries whose answer
lies in O1, beta = mean L2 distance between a translated O1 vector and its true target.

`run_cell.py --reference-method la2m` selects the split of the original VectorMerge code
instead: answers kept out of O_cap and spread evenly over O1 and O2, only O1 translated.
It puts Union and A2M about 13-15 points of R above the published rows (Union about 0.5
of the target model's own recall, against the paper's about 0.3), so the published rows
were not produced with it.

## 5. What reproduces and what does not

### 5.1 The paper's claims

- **LA2M integrates databases across models.** On all 30 cells of Tables 4-9, LA2M is at
  or above A2M on both R and R1, and above Union on R in 29 (the exception is Table 7 /
  SciDocs, 14.0 vs 14.4). LA2M's R is 1.0-2.9 times Union's (median 1.8).
- **Union and A2M match the published rows.** Averaged over the 30 cells, ours are R +2.4
  (Union) and R +3.6, R1 -1.8 (A2M) from the paper.
- **LA2M's own numbers.** Averaged over the 30 cells, our LA2M is within a few points of the
  published rows (R -0.9, R1 +5.9 on average); Table 7 (FastText -> NV-Embed-V2) is the
  one table where ours is clearly lower (R -8.9).
- **Local isometries peak at the intrinsic dimension (Fig. 7).** With the paper's intrinsic
  dimensions (42 / 55 / 110), all three model pairs start at Union level, peak at that
  dimension and decline beyond it.
- **Stratified references beat random ones (Fig. 8).** With the sampling clusters used as
  LA2M's local isometries (about 130 references each, clusters under 100 references on the
  paper's A2M), stratified sampling gives the lower beta in 22 of 24 (dataset, gamma) points
  and the higher RN in 14; random sampling leaves about 12% of the translated documents in
  under-sampled clusters, stratified none. From gamma = 0.8 to 0.2 the stratified beta rises
  by 3-4%.

### 5.2 What does not

Some figures are represented by conceptual diagrams with no data behind them (Figs. 1, 2
and 6) and need no reproduction.

The GW and CCA rows match the paper only under the original VectorMerge split
(`run_cell.py --reference-method la2m`, R -1.7 and -1.6 on average), not under the
paper's split used here (R -10.9 and -12.0): our GW maps the source vectors to near the
origin (beta 1.00, as in the paper), so its R only reflects how many answers stay on the
target side. The A2M_MLP row (trained with a cosine-distance objective) matches the
published R (+2.3 on average) but its R1 is about 0 and its beta 2-11 against the paper's
about 1.00: the objective leaves the norm of the translated vectors free, and under L2
search they end up far from every query. CCA's beta is inflated by reconstructing with
`y_weights_`, and diverges on Table 5 / NFCorpus.

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
    run_figures.sh           Figs. 3b, 5, 7, 8, 9, 10
  code/                      everything that runs inside the container
    Dockerfile, pyproject.toml, uv.lock, .vectormerge/config.yaml, configs/
    src/vectormerge/         the package: LA2M (mapping/strategies/la2m.py), A2M
                             (mapping/strategies/procrustes.py), CCA, GW, MLP, reference
                             splits, clustering and intrinsic dimension, evaluation
    scripts/                 run_cell.py (one table cell), figures.py (Figs. 3b, 5, 7, 9),
                             fig8.py, fig10_repro.py, plot_fig8.py, plot_fig9.py
  data/                      created on first run: downloaded inputs
  results/                   created on first run: everything the experiments write
```

