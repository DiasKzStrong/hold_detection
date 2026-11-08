# YOLO26m training from HoldClassificatore

Functions live in `src/hold_detection/yolo_training.py`, with `scripts/train_yolo26.py`
as the Python entry point and SLURM jobs below. Training is separate from the
[inference-only notebook](../notebooks/DeMoSearch_Pipeline.ipynb), which uses existing checkpoints.
See [the complete wall pipeline](wall-pipeline.md).

## Dataset and model

Use [HoldClassificatore_v3_mix](https://universe.roboflow.com/climbing-holds-classification/holdclassificatore_v3_mix)
or [v2.2](https://universe.roboflow.com/climbing-holds-classification/holdclassificatore_v2.2).
Select a published dataset version and export YOLO **object detection**. YOLOv5 PyTorch / YOLOv8
exports contain `data.yaml`, split image folders and normalized bounding-box TXT labels; YOLO26
accepts this format directly. Preparation accepts a local ZIP/folder. Direct download uses
`holds download-yolo`, with ROBOFLOW_API_KEY from the environment. No credentials are committed.

The default is `yolo26m.pt`, fine-tuned for crimp, jug, pinch, pocket, sloper and volume.
The code preserves the export's class-ID order and normalizes `Slope` to `sloper`.
It checks labels/coordinates, rejects missing label files and exact cross-split duplicate pixels,
preserves source splits and attribution, and writes `dataset_report.json`. Empty TXT files mean
verified negative images. A SHA-256 fingerprint includes class order, image pixels and labels.
Only the standard split-folder layouts are accepted; no remote YAML download directives run.

This is box detection with grip labels, not mask training. SAM3 supplies masks in wall analysis.
These six classes omit undercling/sidepull/opening direction. Those need separate reviewed
attributes. A type prediction cannot reconstruct an invisible lip or measure its depth.

See [data provenance and split cautions](classifier-training.md#data-requirements-and-provenance).
This dataset shares sources with the legacy checkpoint's training data. Pixel hashing cannot
detect resizes/mirrors or neighboring frames. Audit physical wall/session groups before claiming
generalization; keep our Polycam captures outside training for a separate evaluation.

## Python entry point

Run training from a configured CUDA environment or use the SLURM jobs below:

```bash
python scripts/train_yolo26.py prepare /path/to/HoldClassificatore.zip --output data/hold-types
python scripts/train_yolo26.py train --data data/hold-types/data.yaml --output runs/yolo26m_smoke \
  --model yolo26m.pt --epochs 3 --imgsz 1024 --batch 8 --device 0
python scripts/train_yolo26.py evaluate --data data/hold-types/data.yaml --output runs/yolo26m_smoke \
  --split val --device 0
```

For a full experiment, use a new run directory and 100 epochs. Resume applies to an interrupted
run with the original epoch budget, not an already completed three-epoch smoke run. Use the test
split only after selecting the configuration. Pass the resulting best.pt to the inference notebook.

Outputs include `weights/best.pt`, `weights/last.pt`, per-epoch checkpoints,
`training_manifest.json`, `results.csv`, plots and `metrics_val.json` / optional `metrics_test.json`.
The base `yolo26m.pt` checkpoint is trained on COCO; it is not a ready climbing model.
No full HoldClassificatore training or GPU accuracy results have been produced locally.

## NYU Cloud Bursting

Prepare a Python 3.12 environment or your own container overlay with matching CUDA
`torch==2.7.1` / `torchvision==0.22.1` first, then install the training requirements:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r scripts/requirements-yolo26.txt
```

Use the same environment on CPU and GPU nodes. Do not run the root `uv sync` there;
the project's Linux lock selects CPU Torch. Existing `scripts/slurm/container_python.sh`
can wrap an already configured CUDA image/overlay. Do not share a writable overlay across jobs.

Download one dataset ZIP to `/scratch/NetID` or transfer it there once. Example commands
below assume the checkout and dataset are already on the cluster:

```bash
export DEMO_PROJECT=/scratch/NetID/hold_detection
export DEMO_PYTHON=/scratch/NetID/yolo26-env/bin/python
export DEMO_SOURCE=/scratch/NetID/datasets/HoldClassificatore.zip
export DEMO_DATA=/scratch/NetID/datasets/HoldClassificatore-prepared
export DEMO_RUN=/scratch/NetID/hold-runs/yolo26m-1024-seed42

sbatch scripts/slurm/yolo26-prepare.sbatch
# After the CPU preparation job completes successfully:
sbatch scripts/slurm/yolo26-a1002.sbatch
# After training completes:
sbatch scripts/slurm/yolo26-evaluate.sbatch
```

The CPU job uses `n2c48m24`. The GPU job uses account `csci_ga_2271-2026fa`, partition
`c24m170-a100-2`, two A100s and global batch 32 (16 per GPU). It uses native torchrun
so all ranks execute the same checkpoint callbacks. These templates have not been submitted.

For direct download on the CPU node, choose a nonexistent ZIP at DEMO_SOURCE and set
DEMO_ROBOFLOW_VERSION to the published version number plus ROBOFLOW_API_KEY in the environment.
The preparation job downloads, validates and prepares the export, then caches the initialization
checkpoint. No manual conversion or dataset upload from a laptop is needed.

Checkpoints are saved after every completed epoch. USR1/TERM requests a pause; once
`last.pt` exists, all ranks stop together at the next batch callback. The unfinished epoch is
repeated on resume. If preemption occurs before the first saved epoch, no resumable checkpoint
exists yet. Uncatchable termination resumes from the previous completed epoch if available.
The job keeps `--requeue`; set `DEMO_AUTO_REQUEUE=1` to ask SLURM to requeue a checkpointed
pause automatically. Exit 75 / a `PAUSED` marker means paused, not completed.
Rerunning the job selects `last.pt` automatically. A completed run is not retrained.

Evaluate the selected checkpoint in a GPU allocation:

```bash
python scripts/train_yolo26.py evaluate --data /scratch/NetID/datasets/HoldClassificatore-prepared/data.yaml \
  --output /scratch/NetID/hold-runs/yolo26m-1024-seed42 --split val --device 0
```

## Unknown grip types in the downstream algorithm

Always retain a detected hold and its position when its type is uncertain. Export the predicted
label, score, source and review status separately. A YOLO detection score is a joint box/class
confidence, not a six-class probability vector; do not invent probabilities from it.
The teammate's algorithm needs an explicit policy for missing types: a learned unknown-type
cost or a reported cost range, and/or human confirmation before using grip-dependent moves.
An unknown label is not evidence that a hold is unusable. For the final blue-V5 demo, manually
verify ambiguous types. Even a confident RGB guess cannot measure unseen grip geometry.

## Hosted training alternative

Roboflow can train models using credits. Its current [pricing](https://roboflow.com/pricing)
and [usage rates](https://roboflow.com/credits) describe the free credit allowance and GPU
training costs. Check the workspace's available architectures before assuming hosted training
offers the exact YOLO26m settings above. Exporting a YOLO dataset and running this notebook/HPC
experiment gives us explicit versions, model size, settings and local checkpoints.
