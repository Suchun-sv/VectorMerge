#!/usr/bin/env bash
# Figures 3b, 4, 5, 7, 9 and 10. Each figure writes its numbers (JSON) and its plot to
# $RESULTS/figures.
#
#   bash scripts/run_figures.sh            # all, ~3 h (Fig. 7 alone ~1.75 h)
#   bash scripts/run_figures.sh 9 10       # just these
#
# Fig. 10 embeds CIFAR-10 with ResNet-50 / DenseNet-121 and wants a GPU (GPU_FLAG below);
# without one, set GPU_FLAG= and expect it to take a few hours on CPU.
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU_FLAG="${GPU_FLAG:---gpus all}"

SIZES=1,5,10,20,30,42,55,100,150,200,300,500,1000,1500,2000   # Fig. 7 x-axis
KS=1,5,10,20,30,40,50,100,150,200,300,500,1000,1500,2000      # Fig. 9 x-axis
F=/app/figures

fig3b() {   # top-k Jaccard between NV-Embed-V2 and OpenAI-Ada on SciFact, ~1 min
  dk -- scripts/figures.py --figure 3b --dataset scifact
}

fig4() {    # T-SNE of a query and its top-10, FastText vs NV-Embed-V2 on SciFact, ~2 min
  dk -- scripts/fig4_repro.py
}

fig5() {    # #clusters vs alpha / beta; FastText -> NV-Embed-V2 plus five more pairs
  for pair in fast-text:nv-embed gte:nv-embed nv-embed:openai mistral:openai \
              glove:fast-text fast-text:openai; do
    IFS=: read -r s t <<< "$pair"
    dk -- scripts/figures.py --figure 5 --dataset scifact --src "$s" --tgt "$t"
  done
}

fig7() {    # recall@100 vs |X|, the Fig 7 version, ~1.75 h
  dk -- scripts/figures.py --figure 7 --dataset scifact --sizes "$SIZES" --version fig7
}

fig9() {    # alpha and recall@100 vs k, ~15 min
  # alpha panel: every cluster fitted upright, no fallback (the alignment error itself)
  dk -- scripts/figures.py --figure 9 --dataset scifact --ns "$KS" --paper-dhat
  # recall panel: the Fig 9 version
  dk -- scripts/figures.py --figure 9 --dataset scifact --ns "$KS" --version fig9
  dk -- scripts/plot_fig9.py $F/fig9_paperdhat.json $F/fig9_fig9-version.json $F/fig9.png
}

fig10() {   # privacy-preserving image search on CIFAR-10, ~15 min with a GPU
  # shellcheck disable=SC2086
  dk --img $GPU_FLAG -- scripts/fig10_repro.py --stage all
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(3b 4 5 7 9 10)
for f in "${targets[@]}"; do
  echo "=== Fig. $f  $(date +%H:%M:%S)"
  "fig$f"
done
echo "FIGURES DONE"
