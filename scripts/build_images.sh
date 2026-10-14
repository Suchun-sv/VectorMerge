#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
docker build -f "$CODE/Dockerfile" -t "$IMAGE" "$CODE"
docker build -f "$ART/scripts/Dockerfile.img" --build-arg BASE="$IMAGE" -t "$IMAGE_IMG" "$ART/scripts"
docker run --rm --entrypoint python "$IMAGE" -c \
  "import vectormerge, torch; print('vectormerge OK, torch', torch.__version__)"
