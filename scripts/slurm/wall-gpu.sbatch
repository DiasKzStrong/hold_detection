#!/usr/bin/env bash
#SBATCH --job-name=wall-gpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=wall-%x-%j.log
set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT}"
: "${DEMO_RUN:?Set DEMO_RUN}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON}"
mkdir -p "${DEMO_RUN}"
if [[ -f "${DEMO_RUN}/STOP_REQUESTED" ]]; then
  "${DEMO_PYTHON}" -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()' "${DEMO_RUN}/STOP_REQUESTED"
fi
trap 'touch "${DEMO_RUN}/STOP_REQUESTED"' USR1 TERM
srun bash "${DEMO_PROJECT}/scripts/slurm/run_wall_stage.sh" &
worker=$!
while true; do
  status=0
  wait "${worker}" || status=$?
  if ! kill -0 "${worker}" 2>/dev/null; then break; fi
done
if [[ "${status}" == 75 && "${DEMO_AUTO_REQUEUE:-0}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol requeue "${SLURM_JOB_ID}"
fi
exit "${status}"
