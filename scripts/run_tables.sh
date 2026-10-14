#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/env.sh"

METHODS="${METHODS:-la2m procrustes gromov_wasserstein cca nonlinear}"
CELL_TIMEOUT="${CELL_TIMEOUT:-3600}"
LA2M_FLAGS="${LA2M_FLAGS:-}"
GPU_FLAG="${GPU_FLAG:---gpus all}"

PAIRS=("table9:glove:fast-text" "table6:fast-text:openai" "table4:mistral:openai"
       "table7:fast-text:nv-embed" "table5:nv-embed:openai" "table8:gte:nv-embed")
DATASETS=(nfcorpus scifact arguana scidocs fiqa)

flags_for() {
  case "$1" in
    la2m) echo "$LA2M_FLAGS" ;;
    *) echo "" ;;
  esac
}

TABLES="${TABLES:-table4 table5 table6 table7 table8 table9}"

for pair in "${PAIRS[@]}"; do
  IFS=: read -r table src tgt <<< "$pair"
  [[ " $TABLES " == *" $table "* ]] || continue
  for ds in "${DATASETS[@]}"; do
    for m in $METHODS; do
      out="$RESULTS/tables/${table}_${ds}_${m}.json"
      [ -f "$out" ] && continue
      echo "$(date +%H:%M:%S) $table $ds $src->$tgt $m"
      find "$DATA/processed/clusters" -mindepth 1 -delete 2>/dev/null
      name="la2m_${table}_${ds}_${m}_$$"
      gpu=""; [ "$m" = nonlinear ] && gpu="$GPU_FLAG"
      timeout "$CELL_TIMEOUT" bash -c "source '$ART/scripts/env.sh'; dk --name $name $gpu -- scripts/run_cell.py \
        --dataset $ds --source $src --target $tgt --method $m $(flags_for "$m") \
        --out /app/results/tables/${table}_${ds}_${m}.json" 2>&1 | tail -1
      if [ "${PIPESTATUS[0]}" -eq 124 ]; then
        docker rm -f "$name" >/dev/null 2>&1
        echo "  timeout after ${CELL_TIMEOUT}s"
        printf '{"dataset": "%s", "source_model": "%s", "target_model": "%s", "mapping_method": "%s", "status": "timeout", "seconds": %s}\n' \
          "$ds" "$src" "$tgt" "$m" "$CELL_TIMEOUT" > "$out"
      fi
    done
  done
done
echo "TABLES DONE"
