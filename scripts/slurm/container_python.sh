#!/usr/bin/env bash
# Optional executable wrapper for DEMO_PYTHON. Set paths to your own NYU overlay/image.
set -euo pipefail
: "${DEMO_SIF:?Choose an existing image under /share/apps/images}"
: "${DEMO_OVERLAY:?Choose your own prepared overlay on /scratch/NetID}"
DEMO_ENV_SH="${DEMO_ENV_SH:-/ext3/env.sh}"
if command -v singularity >/dev/null 2>&1; then
  container=singularity
elif command -v apptainer >/dev/null 2>&1; then
  container=apptainer
else
  echo "Load your site's Singularity/Apptainer module first" >&2; exit 2
fi
gpu=()
if [[ "${DEMO_CPU_ONLY:-0}" != 1 ]]; then gpu=(--nv); fi
exec "${container}" exec "${gpu[@]}" --bind /scratch \
  --overlay "${DEMO_OVERLAY}:ro" "${DEMO_SIF}" /bin/bash -c \
  'source "$1"; shift; exec python "$@"' bash "${DEMO_ENV_SH}" "$@"
