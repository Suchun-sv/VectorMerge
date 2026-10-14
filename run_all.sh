#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/env.sh

bash scripts/build_images.sh
bash scripts/run_tables.sh
bash scripts/run_figures.sh
python3 scripts/collect_tables.py "$RESULTS/tables" > /dev/null
dk -- scripts/make_report.py --out /app/output/report.pdf
cp "$RESULTS/output/report.pdf" "$RESULTS/report.pdf"
echo
echo "Report:  $RESULTS/report.pdf (tables and figures in the paper's layout)"
echo "Tables:  $RESULTS/tables/summary.md"
echo "Figures: $RESULTS/figures (and figures/fig10)"
