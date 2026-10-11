# Shared settings for the artifact scripts; sourced, not executed.
#
#   ART         this artifact directory
#   CODE        the package and experiment scripts (Docker build context)
#   DATA        downloaded data (default LA2M_artifacts/data); persists across runs (embeddings ~5.3 GB, BEIR ~0.5 GB,
#               CIFAR-10 + image vectors ~0.6 GB)
#   RESULTS     everything the experiments write
#   IMAGE       image for the text experiments, IMAGE_IMG adds torchvision for Fig. 10
#   CPUS, MEM   per-container limits; THREADS for BLAS / torch

ART="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$ART/code"
DATA="${DATA:-$ART/data}"
RESULTS="${RESULTS:-$ART/results}"
# docker -v needs absolute paths (a relative one is taken as a named volume)
mkdir -p "$DATA" "$RESULTS"
DATA="$(cd "$DATA" && pwd)"; RESULTS="$(cd "$RESULTS" && pwd)"
IMAGE="${IMAGE:-la2m-artifact}"
IMAGE_IMG="${IMAGE_IMG:-la2m-artifact-img}"
CPUS="${CPUS:-28}"
MEM="${MEM:-80g}"
THREADS="${THREADS:-16}"
mkdir -p "$RESULTS/tables" "$RESULTS/figures" "$RESULTS/output"

# dk [extra docker args...] -- python args...
# Runs python in the image as the calling user, with the data, code/scripts and the
# results mounted where the code expects them (/app/data, /app/scripts, /app/figures,
# /app/results/tables).
dk() {
  local image="$IMAGE" extra=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do
    case "$1" in
      --img) image="$IMAGE_IMG" ;;
      *) extra+=("$1") ;;
    esac
    shift
  done
  shift
  docker run --rm --memory="$MEM" --cpus="$CPUS" --shm-size=8g \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$DATA:/app/data" \
    -v "$CODE/scripts:/app/scripts:ro" \
    -v "$RESULTS/figures:/app/figures" \
    -v "$RESULTS/tables:/app/results/tables" \
    -v "$RESULTS/output:/app/output" \
    -e TORCH_HOME=/app/data/torch -e HF_HOME=/app/data/hf \
    -e MPLCONFIGDIR=/tmp/mpl -e WANDB_MODE=offline -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e PYTHONUNBUFFERED=1 -e OMP_NUM_THREADS="$THREADS" \
    -e OPENBLAS_NUM_THREADS="$THREADS" -e MKL_NUM_THREADS="$THREADS" \
    "${extra[@]}" --entrypoint python "$image" "$@"
}
