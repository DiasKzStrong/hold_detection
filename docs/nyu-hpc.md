# NYU Fall 2026 Cloud Bursting: two-A100 training

These templates use the course settings supplied by the user:

- Portal: <https://ood.burst.hpc.nyu.edu/>; NYU VPN is required off campus.
- Account: `csci_ga_2271-2026fa` (course allocation: 300 GPU hours per student).
- Requested GPU partition: **`c24m170-a100-2`**, two A100 40 GB GPUs.
- CPU partition: `n2c48m24`.
- Persistent data/checkpoints: `/scratch/NetID`.
- Coursework computation must use **Cloud Bursting, not the Torch cluster**.

No jobs are submitted by repository setup or tests. Account/partition access,
site modules and your Singularity/Conda environment must be configured in NYU.
Two allocated GPUs consume allocation faster than one; no measured speedup or
wall-clock estimate is claimed. Do not leave GPU sessions idle: the course warns
of automatic shutdown after 20 consecutive minutes of low GPU utilization.

## Environment

In OOD, open an authorized terminal or Jupyter terminal. Put the checkout and
labeled dataset on your scratch storage. The CLI requires Python 3.12 and matching
CUDA Torch/torchvision. If using an already activated compatible environment:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[training]'
```

Do not subsequently synchronize the root CPU `uv.lock` into this GPU environment.
Check that the NVIDIA driver supports the chosen CUDA wheels. Dataset preparation
and encoder downloads belong on CPU/transfer resources, not in a long idle GPU allocation.

For the site's overlay approach, select actual available files under
`/share/apps/images` and `/share/apps/overlay-fs-ext3`, following
[NYU's Singularity/Miniconda guide](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda).
Do not invent an image name or modify a shared template. Install dependencies into
your own overlay using the site's documented writable-overlay setup, then use it
read-only in jobs. The optional `container_python.sh` assumes your prepared
environment activation script is `/ext3/env.sh`; override `DEMO_ENV_SH` if different.

```bash
export DEMO_PROJECT="/scratch/${USER}/hold_detection"
export DEMO_SCRATCH="/scratch/${USER}"
export DEMO_SIF='/share/apps/images/SELECT_AN_EXISTING_IMAGE.sif'
export DEMO_OVERLAY="/scratch/${USER}/YOUR_PREPARED_OVERLAY.ext3"
export DEMO_PYTHON="${DEMO_PROJECT}/scripts/slurm/container_python.sh"
```

Alternatively, set `DEMO_PYTHON` to an absolute executable in a prepared GPU
environment. It must work inside Slurm, not just an interactive shell. The wrapper
binds `/scratch`, uses `--nv` for GPU jobs and mounts the user overlay read-only.
Both the project and datasets should be on the bound scratch filesystem.

## 1. Prepare crops on CPU

```bash
export DEMO_SOURCE="/scratch/${USER}/datasets/holds-labeled.zip"
export DEMO_DATA="/scratch/${USER}/datasets/hold-crops"
export DEMO_STAGE=prepare
export DEMO_RUN="${DEMO_DATA}"
sbatch scripts/slurm/cpu.sbatch
```

Wait for completion; inspect `manifest.json`, counts and leakage warnings. If your
Kaggle export lacks YAML, run `prepare-crops --class-names ...` explicitly using
the documented ID order instead. Re-running preparation must use a new output.
For strict wall/scan group checking use `--groups` as described in the training guide.

Warm the shared pretrained caches on CPU before requesting GPU time:

```bash
export DEMO_STAGE=fetch-encoders
export DEMO_RUN="/scratch/${USER}/hold-runs/cache-preparation"
sbatch scripts/slurm/cpu.sbatch
```

The scripts set `TORCH_HOME` and `HF_HUB_CACHE` under your scratch directory.
Public SigLIP does not require the SAM 3 gated-model approval. If compute nodes
have no internet, populate these caches first and set `HF_HUB_OFFLINE=1` for
SigLIP jobs. Keep the recorded resolved model revision for reproducibility.

## 2. ResNet18 on two A100s

```bash
export DEMO_STAGE=resnet18
export DEMO_RUN="/scratch/${USER}/hold-runs/resnet18-v1"
export DEMO_EPOCHS=20
export DEMO_BATCH_SIZE=256
sbatch scripts/slurm/a1002.sbatch
```

The implementation uses single-process PyTorch **DataParallel**, not DDP.
Both visible GPUs are requested and used; batch size is global (256 means about
128 per GPU for full batches). This is a functional baseline, not a claim of
optimal multi-GPU scaling. Model state is saved without `module.` prefixes.
For a first budgeted smoke run, set `DEMO_TIME_BUDGET_SECONDS=300`; exit 75 and
`last.pt` are expected. Then restore it to 6900 and submit the same configuration
to resume. Do not interpret that short run as a trained-model quality estimate.

## 3. SigLIP encoder on two A100s

```bash
export DEMO_STAGE=siglip-features
export DEMO_RUN="/scratch/${USER}/hold-runs/siglip-features-v1"
export DEMO_BATCH_SIZE=128
export DEMO_SIGLIP_REVISION=main
sbatch scripts/slurm/a1002.sbatch
```

Existing completed feature shards are reused if the same job is submitted again.
Do not change extraction batch size/model/dataset in the middle of a cache.
Use a separate directory for a new experiment. The exact Hub commit is stored
in `cache.json`; `main` is not treated as an immutable revision in the report.

## 4. Linear head on CPU

After extraction completes:

```bash
export DEMO_STAGE=siglip-head
export DEMO_FEATURES="/scratch/${USER}/hold-runs/siglip-features-v1"
export DEMO_RUN="/scratch/${USER}/hold-runs/siglip-head-v1"
export DEMO_EPOCHS=30
export DEMO_BATCH_SIZE=512
sbatch scripts/slurm/cpu.sbatch
```

This does not reserve GPUs or rerun SigLIP. Preserve the cache, including its
frozen encoder file, until the final portable `best.pt` has been packaged.

## Monitor and restart

```bash
squeue -u "$USER"
sacct -j YOUR_JOB_ID --format=JobID,State,Elapsed,AllocTRES
```

Logs are written to the directory from which `sbatch` was called; submit from
persistent scratch, not an ephemeral notebook directory. Checkpoints and caches
use `DEMO_RUN` on scratch. Scripts request `--requeue` and USR1 two minutes before
the time limit. Training also pauses around 6,900 seconds and writes a checkpoint.
Exit 75 is an intentional checkpointed pause; rerun the same `sbatch` command
with unchanged training parameters. The script adds `--resume` when `last.pt`
exists. `--requeue` makes jobs eligible; it does not guarantee automatic requeue
on time limit, idle shutdown, failure or any particular site policy.

SIGUSR1/SIGTERM cause training to checkpoint at a safe batch boundary. SIGKILL
cannot be handled; the last periodic checkpoint is used. Feature extraction
keeps completed shards and restarts an unfinished batch. Confirm the saved
`metrics.json` and `best.pt`; a checkpoint alone is not a finished experiment.

Never reuse one output directory for concurrent jobs. For SIGLIP/ResNet comparison,
keep one dataset manifest and independent run directories. Stage completed artifacts
to longer-term storage according to NYU scratch retention rules.
