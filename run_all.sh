#!/usr/bin/env bash
# ~16 h on the machine in README (Hardware); FAST=1 restricts the tables to the headline
# rows (LA2M, A2M, and Union, which every cell reports) and takes ~5 h.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh

bash scripts/build_images.sh
if [ "${FAST:-0}" = 1 ]; then
  METHODS="la2m procrustes" bash scripts/run_tables.sh
else
  bash scripts/run_tables.sh
fi
bash scripts/run_figures.sh
python3 scripts/collect_tables.py "$RESULTS/tables" > /dev/null
echo
echo "Tables:  $RESULTS/tables/summary.md"
echo "Figures: $RESULTS/figures (and figures/fig10)"
