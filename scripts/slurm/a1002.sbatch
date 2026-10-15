#!/usr/bin/env bash
# NYU Fall 2026 Cloud Bursting only; never submit coursework to Torch.
#SBATCH --job-name=hold-classifier
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=c24m170-a100-2
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=holds-%x-%j.log

set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT to the repository checkout on /scratch/NetID}"
: "${DEMO_DATA:?Set DEMO_DATA to the prepared crop directory}"
: "${DEMO_PYTHON:?Set DEMO_PYTHON to an absolute Python or container-wrapper executable}"
: "${DEMO_RUN:?Set DEMO_RUN to a persistent experiment or feature-cache directory}"
DEMO_STAGE="${DEMO_STAGE:-resnet18}"
DEMO_SCRATCH="${DEMO_SCRATCH:-/scratch/${USER}}"
export PYTHONPATH="${DEMO_PROJECT}/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TORCH_HOME="${DEMO_SCRATCH}/hold-cache/torch"
export HF_HUB_CACHE="${DEMO_SCRATCH}/hold-cache/huggingface/hub"
export HF_HUB_DISABLE_XET=1
export MPLCONFIGDIR="${DEMO_SCRATCH}/hold-cache/matplotlib"
export OMP_NUM_THREADS=4
cd "${DEMO_PROJECT}"
mkdir -p "${TORCH_HOME}" "${HF_HUB_CACHE}" "${MPLCONFIGDIR}"
echo "Cloud Bursting stage=${DEMO_STAGE}; requested 2 A100 GPUs; checkpoint directory=${DEMO_RUN}"
nvidia-smi --query-gpu=name,memory.total --format=csv

case "${DEMO_STAGE}" in
  resnet18)
    resume=()
    if [[ -f "${DEMO_RUN}/last.pt" ]]; then resume=(--resume); fi
    srun "${DEMO_PYTHON}" -m hold_detection.cli train-classifier \
      --architecture resnet18 --dataset "${DEMO_DATA}" --output "${DEMO_RUN}" \
      --epochs "${DEMO_EPOCHS:-20}" --batch-size "${DEMO_BATCH_SIZE:-256}" \
      --workers 8 --threads 4 --gpus 2 --device cuda:0 \
      --checkpoint-every 100 --time-budget-seconds "${DEMO_TIME_BUDGET_SECONDS:-6900}" "${resume[@]}"
    ;;
  siglip-features)
    # Existing complete shards are reused after preemption/requeue.
    srun "${DEMO_PYTHON}" -m hold_detection.cli extract-features \
      --dataset "${DEMO_DATA}" --output "${DEMO_RUN}" \
      --batch-size "${DEMO_BATCH_SIZE:-128}" --workers 8 --threads 4 \
      --gpus 2 --device cuda:0 --revision "${DEMO_SIGLIP_REVISION:-main}"
    ;;
  *) echo "DEMO_STAGE must be resnet18 or siglip-features" >&2; exit 2 ;;
esac
