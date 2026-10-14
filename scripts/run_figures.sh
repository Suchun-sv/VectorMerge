#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU_FLAG="${GPU_FLAG:---gpus all}"

SIZES=1,5,10,20,30,42,55,100,150,200,300,500,1000,1500,2000
KS=1,5,10,20,30,40,50,100,150,200,300,500,1000,1500,2000
F=/app/figures

fig3b() {
  dk -- scripts/figures.py --figure 3b --dataset scifact
}

fig5() {
  for pair in fast-text:nv-embed gte:nv-embed nv-embed:openai mistral:openai \
              glove:fast-text fast-text:openai; do
    IFS=: read -r s t <<< "$pair"
    dk -- scripts/figures.py --figure 5 --dataset scifact --src "$s" --tgt "$t"
  done
}

fig7() {
  dk -- scripts/figures.py --figure 7 --dataset scifact --sizes "$SIZES" --version fig7
}

fig8() {
  dk -- scripts/fig8.py --out $F/fig8
  dk -- scripts/plot_fig8.py $F/fig8 $F/fig8.png
}

fig9() {
  dk -- scripts/figures.py --figure 9 --dataset scifact --ns "$KS" --paper-dhat
  dk -- scripts/figures.py --figure 9 --dataset scifact --ns "$KS" --version fig9
  dk -- scripts/plot_fig9.py $F/fig9_paperdhat.json $F/fig9_fig9-version.json $F/fig9.png
}

fig10() {
  dk --img $GPU_FLAG -- scripts/fig10_repro.py --stage all
}

targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(3b 5 7 8 9 10)
for f in "${targets[@]}"; do
  echo "=== Fig. $f  $(date +%H:%M:%S)"
  "fig$f"
done
echo "FIGURES DONE"
