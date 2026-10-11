#!/usr/bin/env bash
# Tables 4-9 (and Table 1, a subset of them): every (model pair, dataset, method) cell,
# one container per cell, results as JSON in $RESULTS/tables. Resumable: finished cells
# are skipped.
#
#   bash scripts/run_tables.sh                 # all methods, ~13 h (CCA alone ~10 h)
#   METHODS="la2m procrustes" bash scripts/run_tables.sh   # the headline rows, ~1 h
#   TABLES=table9 bash scripts/run_tables.sh                # one table
#
# Methods and the paper's row names:
#   la2m               LA2M          package default (local isometries, d_hat estimated)
#   procrustes         A2M
#   gromov_wasserstein GW
#   cca                CCA
#   nonlinear          A2M_MLP
# Union is reported by every cell (direct_concat), so it needs no run of its own.
set -uo pipefail
source "$(dirname "$0")/env.sh"

METHODS="${METHODS:-la2m procrustes gromov_wasserstein cca nonlinear}"
CELL_TIMEOUT="${CELL_TIMEOUT:-3600}"
LA2M_FLAGS="${LA2M_FLAGS:-}"               # e.g. --transpose-local, see README

# table:source:target, cheapest first
PAIRS=("table9:glove:fast-text" "table6:fast-text:openai" "table4:mistral:openai"
       "table7:fast-text:nv-embed" "table5:nv-embed:openai" "table8:gte:nv-embed")
DATASETS=(nfcorpus scifact arguana scidocs fiqa)

flags_for() {
  case "$1" in
    la2m) echo "$LA2M_FLAGS" ;;
    *) echo "" ;;
  esac
}

TABLES="${TABLES:-table4 table5 table6 table7 table8 table9}"   # e.g. TABLES=table9

for pair in "${PAIRS[@]}"; do
  IFS=: read -r table src tgt <<< "$pair"
  [[ " $TABLES " == *" $table "* ]] || continue
  for ds in "${DATASETS[@]}"; do
    # the published Mistral query vectors for SciFact have 320 rows against 300 queries,
    # so this column cannot be evaluated (README, "Known gaps")
    [ "$table/$ds" = "table4/scifact" ] && continue
    for m in $METHODS; do
      out="$RESULTS/tables/${table}_${ds}_${m}.json"
      [ -f "$out" ] && continue
      echo "$(date +%H:%M:%S) $table $ds $src->$tgt $m"
      # the cluster cache key omits the target model; clear it so pairs cannot collide
      find "$DATA/processed/clusters" -mindepth 1 -delete 2>/dev/null
      name="la2m_${table}_${ds}_${m}_$$"
      timeout "$CELL_TIMEOUT" bash -c "source '$ART/scripts/env.sh'; dk --name $name -- scripts/run_cell.py \
        --dataset $ds --source $src --target $tgt --method $m $(flags_for "$m") \
        --out /app/results/tables/${table}_${ds}_${m}.json" 2>&1 | tail -1
      if [ "${PIPESTATUS[0]}" -eq 124 ]; then
        # killing the client leaves the container running; stop it explicitly
        docker rm -f "$name" >/dev/null 2>&1
        echo "  timeout after ${CELL_TIMEOUT}s"
        printf '{"dataset": "%s", "source_model": "%s", "target_model": "%s", "mapping_method": "%s", "status": "timeout", "seconds": %s}\n' \
          "$ds" "$src" "$tgt" "$m" "$CELL_TIMEOUT" > "$out"
      fi
    done
  done
done
echo "TABLES DONE"
