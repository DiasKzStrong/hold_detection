# DeMoSearch climbing wall perception

A research pipeline for finding climbing holds and volumes, proposing routes
from mask colors, and preparing observations for multi-view 3D association. It
uses the trained YOLO11s checkpoint from
[Marcuss2106's project](https://github.com/Marcuss2106/indoor-climbing-gym-hold-classification),
with an optional separate crop CNN from the same project and an optional
SAM 2.1 adapter for box-prompted instance masks.

**SAM 3:** an independent text-prompted detection + mask backend is now available.
See [SAM 3 setup and viewing instructions](docs/sam3.md). Its official code was
installed, but this cloud machine lacks CUDA and Hugging Face weights are blocked
by its proxy; actual SAM 3 predictions are not yet verified.

This is an inference baseline, not a validated route or reconstruction system.
See [checkpoint provenance and limitations](docs/model-provenance.md).

## Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_SAM3_Colab.ipynb).

The notebook compares **YOLO11s detection** with **SAM 3 instance masks** on the
same photos. It exports mask-color route candidates, a draft `scene.json`, route
review images, diagnostics, and an offline report download.
Start with its default of three photos. YOLO runs first, even without GPU/HF access.

Select a GPU runtime for SAM; the notebook creates its own Python 3.12 model
environment, including when Colab's kernel uses Python 3.13. It does not import or
modify the kernel's PyTorch. The first setup downloads pinned CPU/CUDA wheels and
can take several minutes. SAM's official source revision is pinned;
approved access to `facebook/sam3` and `HF_TOKEN` in Colab Secrets (or a hidden
prompt) are required. The notebook checks out this public repository and records
the exact commit. Photos, Polycam raw-data ZIPs and Drive inputs are supported.

Upload COCO ground truth to obtain precision/recall/F1, box AP, SAM mask AP/IoU
and type confusion matrices. Without labels it reports diagnostics, **not accuracy**.
See [notebook controls and evaluation protocol](docs/colab.md).
Actual SAM GPU inference has not been validated on this CPU-only development machine.

Regenerate the notebook after notebook edits with
`python scripts/build_colab_notebook.py`; it never embeds photos, weights or tokens.

## Legacy grip-classifier research

Grip classification is outside the current route/spatial pipeline. The preserved
legacy experiment supports **ImageNet ResNet18 fine-tuning** and **frozen SigLIP +
linear head**, with labeled YOLO/COCO crop preparation, cached embeddings,
validation-selected checkpoints, restart and test reports. Artifacts plug into
SAM 3 using `--classifier trained --classifier-weights /path/to/best.pt`.

- [Training guide](docs/classifier-training.md)
- [Training notebook in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Classifier_Training.ipynb)
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

## Find holds and propose color routes

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
`scene.json` is the future planner interface. Its `physical_holds` list remains
empty for ordinary photos. For a prepared Polycam capture and instance masks, it
contains multi-view RGB-D hold candidates with metric position, size, and
protrusion in the unaligned capture coordinate frame. GLB registration remains
explicitly `not_registered`. See [the demo guide](docs/demo.md).

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

The prepared directory retains the GLB beside `capture/`. `scene.json` records
its path but keeps `glb_registration = not_registered` until a verified transform
is available. Multi-view association uses sampled 3D mask points, not only 2D box
or centroid distance. Color groups are route candidates, not guaranteed unique
routes: the dataset contains two green routes on wall A.

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

The importer preserves original and corrected RGB/camera pairs, depth maps,
confidence maps and mesh metadata. It rejects unsafe ZIP paths and existing
output directories. Point inference at `corrected_images`, not the export root
or a directory containing depth images. Processing exported data makes no
requests to Polycam. There is no reconstruction, RGB-D fusion, multi-view hold
identity or metric-depth claim yet. Color-derived `route_candidate_id` values
belong to frame observations; `physical_holds` remains empty.

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
