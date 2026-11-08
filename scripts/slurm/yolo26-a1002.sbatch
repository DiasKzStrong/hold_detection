#!/usr/bin/env bash
# NYU Cloud Bursting; native torchrun keeps notebook callbacks on both ranks.
#SBATCH --job-name=holds-yolo26m
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
#SBATCH --output=holds-%x-%j.log

set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT to the repository checkout on /scratch/NetID}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON to an absolute YOLO26 Python or container wrapper}"
: "${DEMO_DATA:?Set DEMO_DATA to the prepared YOLO dataset directory}"
: "${DEMO_RUN:?Set DEMO_RUN to a persistent experiment directory}"
DEMO_SCRATCH="${DEMO_SCRATCH:-/scratch/${USER}}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export YOLO_CONFIG_DIR="${DEMO_SCRATCH}/hold-cache/yolo26"
export MPLCONFIGDIR="${DEMO_SCRATCH}/hold-cache/matplotlib"
export OMP_NUM_THREADS=4
cd "${DEMO_PROJECT}"
mkdir -p "${DEMO_RUN}" "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}"
stop_file="${DEMO_RUN}/STOP_REQUESTED"
# Clear only the resolved previous stop marker when this job is explicitly resumed/requeued.
if [[ -f "${stop_file}" ]]; then
  "${DEMO_PYTHON}" -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()' "${stop_file}"
fi
trap 'touch "${stop_file}"' USR1 TERM
resume=()
if [[ -f "${DEMO_RUN}/weights/last.pt" ]]; then resume=(--resume); fi
if [[ -f "${DEMO_RUN}/TRAINING_COMPLETE" ]]; then
  echo "Training already completed. Use the evaluate command or choose a new DEMO_RUN."
  exit 0
fi
nvidia-smi --query-gpu=name,memory.total --format=csv
# Global batch size is divided across two GPUs. Each rank runs this same generated script.
srun "${DEMO_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_yolo26.py train --data "${DEMO_DATA}/data.yaml" --output "${DEMO_RUN}" \
  --model "${DEMO_MODEL:-yolo26m.pt}" --epochs "${DEMO_EPOCHS:-100}" \
  --imgsz "${DEMO_IMAGE_SIZE:-1024}" --batch "${DEMO_BATCH_SIZE:-32}" \
  --workers 4 --device 0,1 --stop-file "${stop_file}" "${resume[@]}" &
training_pid=$!
status=0
# A signal interrupts wait; continue waiting until the checkpoint-aware workers exit.
while true; do
  status=0
  wait "${training_pid}" || status=$?
  if ! kill -0 "${training_pid}" 2>/dev/null; then break; fi
done
if [[ -f "${DEMO_RUN}/PAUSED" && -f "${DEMO_RUN}/weights/last.pt" ]]; then
  echo "Paused with a resumable completed-epoch checkpoint."
  if [[ "${DEMO_AUTO_REQUEUE:-0}" == 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol requeue "${SLURM_JOB_ID}"
  fi
  exit 75
fi
exit "${status}"
