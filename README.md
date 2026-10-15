# DeMoSearch hold detection

A local baseline for finding climbing holds/volumes and predicting their types
from wall photographs. It uses the trained YOLO11s checkpoint from
[Marcuss2106's project](https://github.com/Marcuss2106/indoor-climbing-gym-hold-classification),
with an optional separate crop CNN from the same project and an optional
SAM 2.1 adapter for box-prompted instance masks.

**SAM 3:** an independent text-prompted detection + mask backend is now available.
See [SAM 3 setup and viewing instructions](docs/sam3.md). Its official code was
installed, but this cloud machine lacks CUDA and Hugging Face weights are blocked
by its proxy; actual SAM 3 predictions are not yet verified.

This is an inference baseline, not a validated classifier for our gym.
See [checkpoint provenance and limitations](docs/model-provenance.md).

## Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_SAM3_Colab.ipynb).

Upload [DeMoSearch_SAM3_Colab.ipynb](notebooks/DeMoSearch_SAM3_Colab.ipynb) to
<https://colab.research.google.com/> and enable a GPU runtime. It embeds this
pipeline's source, installs official SAM 3 at a pinned revision, reads `HF_TOKEN`
from Colab Secrets or a hidden prompt, accepts photos/Polycam ZIP/Drive, previews
rotation, runs segmentation plus optional classification, and downloads an
offline results gallery. Start with its default of one image. Approved access
to `facebook/sam3` is required. Actual GPU inference has not been validated here.

Regenerate the notebook after package edits with
`python scripts/build_colab_notebook.py`; it never embeds photos, weights or tokens.

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

## Find holds and predict types

```bash
uv run holds predict path/to/wall.jpg --output runs/wall-001
uv run holds predict path/to/photos --output runs/batch-001 --device cpu
```

Output directories must be new; existing results are never silently overwritten.
Open the run's `index.html` in a browser to view all annotated photos and expand
their crops/type predictions. Copy the whole run folder to view it locally.
Each frame gets `annotated.jpg`, `holds.json`, and individual `crops/`.
`summary.json` records settings, checkpoint hashes, timings and per-frame counts.
IDs are frame-local; multi-view tracking and 3D registration are not implemented.

Generate a gallery for an older run with `uv run holds gallery runs/<run-name>`.

The default classification uses the detector's joint detection/class score.
For the separate crop classifier:

```bash
uv run holds predict path/to/wall.jpg --output runs/cnn-001 --classifier cnn
```

The output retains both the detector label and the crop classifier's six scores.
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
requests to Polycam. There is no reconstruction, RGB-D fusion, route extraction
or metric-depth claim yet; `route_id` and `position_3d` remain null.

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
including holes; mask-derived median RGB is exported separately from route IDs.
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

To evaluate the crop CNN on your own labels (one integer class ID per TXT file,
in the upstream class order):

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

Next: annotate holds, types, and non-hold false positives on selected frames;
split by physical wall/scan rather than random crops; compare detector labels
with the crop CNN and then a SigLIP 2 baseline. Evaluate detection recall and
per-class F1 before relying on labels for the planner. Keep ambiguous grip types
explicit. Two green routes on wallA_1 require more than color to distinguish.
