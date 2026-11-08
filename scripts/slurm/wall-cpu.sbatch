#!/usr/bin/env bash
#SBATCH --job-name=wall-cpu
#SBATCH --account=csci_ga_2271-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=wall-%x-%j.log
set -euo pipefail
: "${DEMO_PROJECT:?Set DEMO_PROJECT}"
export DEMO_CPU_ONLY=1
srun bash "${DEMO_PROJECT}/scripts/slurm/run_wall_stage.sh"
