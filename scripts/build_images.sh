#!/usr/bin/env bash
# Build the two images: la2m-artifact (text experiments) and la2m-artifact-img
# (adds torchvision for the image case study, Fig. 10). About 5 minutes on first build.
set -euo pipefail
source "$(dirname "$0")/env.sh"
docker build -f "$CODE/Dockerfile" -t "$IMAGE" "$CODE"
docker build -f "$ART/scripts/Dockerfile.img" --build-arg BASE="$IMAGE" -t "$IMAGE_IMG" "$ART/scripts"
docker run --rm --entrypoint python "$IMAGE" -c \
  "import vectormerge, torch; print('vectormerge OK, torch', torch.__version__)"
