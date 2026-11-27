# DeMoSearch climbing wall perception

DeMoSearch analyzes a selected section of a climbing wall. **2D** takes one photo,
with optional crop. **3D** takes overlapping photos or video and reconstructs the
section with COLMAP. Both modes use SAM 3 for hold masks, optional YOLO26 for hold
types, and measured hold colors. Both export `scene.json` for the downstream graph algorithm.
3D also exports a GLB with hold coordinates and correspondence evidence.

[Open the inference notebook in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Pipeline.ipynb).
Select a GPU runtime, set `MODE` and the input paths, and provide ready checkpoints.
See [the module map, CLI, and SLURM instructions](docs/wall-pipeline.md).

## Mobile demo API

Upload a photo or video, poll analysis status, and download the scene and 3D wall.
See the [HTTP API setup and contract](docs/demo-api.md).

## Comparison experiment

The [YOLO11s and SAM 3 notebook](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/experiments/DeMoSearch_SAM3_Colab.ipynb)
compares models on the same photos, with optional COCO metrics and an offline report.
See the [experiment controls](docs/colab.md), [SAM 3 setup](docs/sam3.md), and
[checkpoint provenance](docs/model-provenance.md).

## YOLO26m training

Train the six-class RGB hold-type detector on a HoldClassificatore YOLO export,
with dataset checks, validation reports and resumable checkpoints. Training runs through
`scripts/train_yolo26.py` or two-A100 NYU Cloud Bursting jobs, separately from the inference notebook.
See [training and HPC instructions](docs/yolo26-training.md).

## Other grip-classifier experiments

The earlier crop-classification experiment supports **ImageNet ResNet18 fine-tuning** and **frozen SigLIP +
linear head**, with labeled YOLO/COCO crop preparation, cached embeddings,
validation-selected checkpoints, restart and test reports. Artifacts plug into
SAM 3 using `--classifier trained --classifier-weights /path/to/best.pt`.

- [Training guide](docs/classifier-training.md)
- [Training notebook in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/experiments/DeMoSearch_Classifier_Training.ipynb)
- [NYU Cloud Bursting jobs: two A100s](docs/nyu-hpc.md), account `csci_ga_2271-2026fa`,
  partition `c24m170-a100-2`; data preparation and linear-head training use CPU jobs.

Unlabeled gym photos are inference inputs, not supervised training labels.
Full-dataset training and actual NYU multi-GPU jobs have not been executed here.

## Install

Python 3.10–3.13. From this checkout (cloud tasks already have isolated checkouts;
there is no need to create another worktree):

```bash
uv sync --frozen --extra dev
uv run holds fetch-models
```

The lock uses CPU PyTorch wheels on Linux/Windows so inference works without a
GPU or a large CUDA download. macOS uses the PyPI wheels. The checkpoint URLs
are pinned to a Git revision and downloads are checked against SHA-256 hashes.
No Kaggle credentials are required for these inference checkpoints.

If `uv` cannot write its default cache in a cloud machine, prefix its commands
with `UV_CACHE_DIR=/workspace/.cache/uv`.

For GPU development, install the appropriate official PyTorch/torchvision
2.7.1/0.22.1 wheels into `.venv` using `uv pip`; use `.venv/bin/holds` directly
afterward, because another `uv sync` restores the locked CPU environment.

## Legacy prediction and Polycam commands

```bash
uv run holds predict path/to/wall.jpg --output runs/wall-001
uv run holds predict path/to/photos --output runs/batch-001 --device cpu
```

Output directories must be new; existing results are never silently overwritten.
Open the run's `index.html` in a browser to review detections, masks, color-route
groups, and crops. Copy the whole run folder to view it locally. Every run gets
`scene.json`; each frame gets `annotated.jpg`, `routes.jpg`, `holds.json`, and
individual `crops/`. Segmentation runs additionally export lossless masks.
`summary.json` records settings, checkpoint hashes, timings and per-frame counts.
The legacy scene uses the draft schema. Its `physical_holds` list remains
empty for ordinary photos. For a prepared Polycam capture and instance masks, it
contains multi-view RGB-D hold candidates with metric position, size, and
protrusion in the capture coordinate frame. With a complete wall bundle, a
quality-gated registration stage also attempts to match those holds to the GLB;
the status is explicitly `registered`, `rejected`, `failed`, or `not_registered`.
See [the demo guide](docs/demo.md).

Generate a gallery for an older run with `uv run holds gallery runs/<run-name>`.

### Complete wall bundles

The preferred input is one wall directory containing a Polycam raw-data ZIP (or
an already extracted raw-data directory), one GLB, optional OBJ backup, route
photos, and `route_notes.txt`. ZIP filenames do not need a `.zip` suffix.

```bash
uv run holds inspect-wall-bundle /path/to/wallC_overhang_right
uv run holds prepare-wall-bundle /path/to/wallC_overhang_right \
  --output /tmp/wallC-prepared
uv run holds sam3 /tmp/wallC-prepared/capture/keyframes/corrected_images \
  --output runs/wallC-sam3 --classifier none
```

The prepared directory retains the GLB beside `capture/`. When physical holds are
available, the scene builder estimates a rigid capture-to-GLB transform using
PCA-initialized trimmed ICP, reports its residual and inlier ratios, and rejects
weak registration. Accepted matches add GLB position, normal, nearest mesh anchor,
and point-to-mesh distance to every physical hold. Multi-view association uses
sampled 3D mask points, not only 2D box or centroid distance. Color groups are
route candidates, not guaranteed unique routes: the dataset contains two green
routes on wall A.

Grip classification is disabled by default. The old detector labels can still be
requested explicitly for legacy experiments:

```bash
uv run holds predict path/to/wall.jpg --output runs/legacy-types \
  --classifier detector
```

Legacy classification output retains detector labels and optional crop-classifier scores.
Disagreement is flagged for review. Neither model's score is a calibrated
probability of correctness. Predictions below `--min-type-score` (default 0.5)
are reported as `unknown`. Volumes have no grip type. The class vocabulary is
`crimp`, `jug`, `pinch`, `pocket`, `sloper`, and `volume`.

Useful settings:

```bash
# Higher recall, including more false positives:
uv run holds predict wall.jpg --output runs/recall-001 --conf 0.25
# Full-image detection plus overlapping tiles for small holds:
uv run holds predict wall.jpg --output runs/tiles-001 --tile-size 640 --overlap 0.25
```

`--rotate 90` rotates pixels clockwise before inference. All exported boxes and
masks refer to the processed image, not the original camera coordinates.
Rotation is recorded in JSON; there is no automatic EXIF rotation. Do not use
these coordinates with Polycam intrinsics without first accounting for rotation.

## Polycam exports

```bash
uv run holds inspect-polycam /path/to/wallA_1_raw_data.zip
uv run holds prepare-polycam /path/to/wallA_1_raw_data.zip --output data/wallA_1
uv run holds predict data/wallA_1/keyframes/corrected_images \
  --output runs/wallA_1 --device cpu --rotate 90
```

The importer preserves RGB/camera pairs, depth, confidence maps, and mesh metadata.
Point inference at `keyframes/corrected_images`. For RGB-D fusion, use SAM 3 masks
with the prepared capture. The COLMAP workflow accepts overlapping RGB images
or a video through `analyze-wall --mode 3d`.

## Optional SAM 2.1 masks

Install the official implementation at a pinned revision, without a CUDA build
for a CPU machine:

```bash
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation \
  'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
curl --fail --location \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt \
  --output models/sam2.1_hiera_tiny.pt
.venv/bin/holds predict wall.jpg --output runs/sam-001 \
  --sam2-checkpoint models/sam2.1_hiera_tiny.pt
```

Use `.venv/bin/holds` after installing SAM: plain `uv run` can synchronize away
packages not in the base lock. Mask PNGs preserve the entire binary mask,
including holes; mask-derived median RGB is exported separately from route
candidate IDs.
SAM refines boxes; it cannot recover a hold the detector never proposed.

During the initial cloud run, the official checkpoint host returned HTTP 403
even after an escalated connectivity check. The adapter is implemented, but
actual SAM inference has **not** been verified. To run it here, environment
network access must permit `dl.fbaipublicfiles.com`, or provide the official
checkpoint as a local file. Box-only inference works without this dependency.

## Validation and next experiments

```bash
uv run pytest
```

Compare completed runs in a portable HTML/CSV/JSON report:

```bash
uv run --extra dev holds benchmark \
  --run YOLO runs/yolo-test --threshold YOLO 0.35 \
  --run SAM3 runs/sam3-test --threshold SAM3 0.5 \
  --annotations path/to/coco.json --gt-coordinates source \
  --output runs/comparison-test
```

Both runs must cover the same input files and rotation. Omit `--annotations` for
unlabeled diagnostics. Save low-confidence proposals during inference (e.g.
`--conf 0.05`) for AP; sliders cannot restore proposals discarded earlier.
Box AP is class-agnostic; the explicit maxDets=500 differs from standard COCO's 100.

The legacy grip-type research remains available but is not part of the current
route-perception scope. To evaluate its crop CNN on old labels:

```bash
uv run holds evaluate-crops path/to/test_split --output runs/cnn-evaluation.json
```

The split must contain `images/<name>.jpg` and `labels/<name>.txt`. The report
includes a confusion matrix, accuracy and per-class precision/recall/F1. It
does not accept detection-format YOLO box labels as crop class labels.

Tests cover duplicate removal, nested holds on volumes, tile coverage, review
flags, artifact export, rotation, empty predictions, output preservation and
safe Polycam import. Real model inference is checked separately on our exported
RGB frames. Unlabeled predictions do not establish accuracy.

Next: annotate hold masks, volumes, color groups, and non-hold false positives;
split by physical wall/capture session rather than random frames; compare SAM 3
with a trained YOLO segmentation model. Evaluate detection recall and mask AP
before multi-view identity and 3D association. Two green routes on wallA_1
require human review or geometry to distinguish.
