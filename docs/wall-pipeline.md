# Wall perception: Python modules, CLI and SLURM

| Module under src/hold_detection | Responsibility |
| --- | --- |
| yolo_training.py | Roboflow download, dataset validation, YOLO26 training/evaluation |
| capture.py | Oriented and bounded frames from one photo, photos/ZIP or video |
| reconstruction.py | CUDA COLMAP SfM/MVS, camera/depth parsing, scale and trimesh GLB export |
| wall_pipeline.py | Resumable per-frame SAM3 subprocesses and YOLO type assignment |
| wall_scene.py | Multiview fusion, spatial estimates and common scene.json export |
| correspondence.py | Matching to named annotated GLB holds with explicit frame alignment |
| wall_commands.py | CLI arguments for these functions |

Notebooks are optional inference clients. The complete inference notebook is edited directly
as an .ipynb and uses existing SAM3/hold-type checkpoints; it contains no training steps.
`scripts/train_yolo26.py` is a thin library entry point.
[Complete inference in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Pipeline.ipynb).
The older SAM3 comparison notebook remains the legacy YOLO11/Polycam comparison.

The inference notebook has separate `INPUT_PATH_2D` and `INPUT_PATH_3D` settings.
The default 3D preset stages 15 overlapping RGB keyframes of the Blue V5 section
from `wallC_raw_data.zip`. For a new wall section, point `INPUT_PATH_3D` to its own
photo folder, ZIP, or video. The notebook can show saved masks next to each photo,
color them by type or group, and display the generated GLB with hold markers.
The same review ID identifies a hold on the mesh and in its supporting frames.

## Input and output

2D accepts exactly one photo (including iPhone HEIC) and exports masks, IDs, normalized 2D centers/sizes, color groups
and optional six-class RGB type proposals. 3D accepts ordinary overlapping photos or a video,
constructs a local wall through COLMAP, segments registered **undistorted images** and lifts masks
using their stereo depth maps. Repeated evidence is fused through color consistency and 3D
sample overlap. Single-view and unmapped holds remain in the output for review.

Enable `USE_2D_CROP` in the inference notebook to select the wall section using a
rectangle and preview the crop. The `prepare-capture --crop-xyxy` CLI option accepts
normalized left, top, right, and bottom edges on the EXIF-oriented source.
The 2D coordinates refer to the prepared crop and `wall.input_crop` records its
pixel bounds in the oriented source.

Both modes return schema 3.0 `scene.json` with identical hold keys; `wall_scene.json` is an
identical compatibility copy. No route planner is included. Image coordinates have top-left
origin, x right and y down. COLMAP's world axes are arbitrary. The 3D scene exporter
estimates `wall.up_vector` from the registered camera poses and records the result in
`wall.up_orientation`. Upright views with enough variation in camera direction can
provide a usable estimate. Ambiguous captures return `manual_required`.

The Colab picker accepts two clicks on a known vertical line in an undistorted
registered image. The CLI accepts the same reference with `export-wall --up-reference`.
An example reference file is
`{"image":"frame_00003.jpg","bottom_px":[320,550],"top_px":[320,180]}`.
Both points need valid, locally consistent stereo depth. This export uses
the existing reconstruction and inference files.

Use `height_3d` to order holds from lower to higher. It is a projection onto the up
vector relative to the reconstruction origin. `height_m` is the same projection
with metric scale applied. Heights are null when the hold is unmapped or vertical
could not be resolved.

```text
RUN/capture/capture.json
RUN/reconstruction/reconstruction.json   # 3D only
RUN/reconstruction/wall.glb              # 3D only
RUN/frame_batches/frame_00000/...        # completed model subprocess checkpoints
RUN/inference/index.html                # mask/color/type review gallery
RUN/inference/FRAME/holds.json
RUN/logs/sam3_00000.log
RUN/scene.json
```

SAM3 uses a fresh process for each frame to release accumulated CPU/GPU memory. Prepared
images have a maximum edge of 1600px; video/photos are sampled to at most 48 frames by default.
Fewer than three overlapping views are rejected in 3D mode. More views, parallax and sharp
images are needed for a useful reconstruction; three views are only a minimum input check.

## Runtime

Use Python 3.12 with CUDA Torch 2.7.1 / torchvision 0.22.1, ultralytics 8.4.207, the project
and pinned official SAM3 dependencies from [SAM3 setup](sam3.md). The notebook runtime helper
can prepare this Python environment. HPC can use an existing CUDA Python or container wrapper;
see [NYU runtime setup](nyu-hpc.md). Do not replace its CUDA wheels with the root CPU uv lock.

3D requires a **CUDA-enabled COLMAP executable**. On Linux x86-64 Colab, the
reconstruction cell installs the pinned conda-forge CUDA 12.6 CLI in a private
prefix on first use. For a local or HPC setup, set DEMO_COLMAP if your CUDA COLMAP
is outside PATH. Installing pycolmap alone does not supply this CLI.
Consult [COLMAP installation](https://colmap.github.io/install.html) and
[CLI documentation](https://colmap.github.io/cli.html). Check `colmap automatic_reconstructor -h`
before a full run. Headless jobs set `QT_QPA_PLATFORM=offscreen`; no unverified cluster module
name is assumed. Download approved SAM3 weights before inference and train the YOLO checkpoint
separately. No GPU job has been submitted by the development agent.

For 3D lifting and calibration the exporter prefers geometric stereo depth. It
uses photometric depth when that is the output of the selected COLMAP preset.

## One CLI command

```bash
holds analyze-wall /path/to/wall.jpg --mode 2d --output runs/photo_demo \
  --sam-weights models/sam3/sam3.pt --type-weights /path/to/yolo26/weights/best.pt

holds analyze-wall /path/to/wall.mov --mode 3d --output runs/video_demo \
  --sam-weights models/sam3/sam3.pt --type-weights /path/to/yolo26/weights/best.pt
```

The YOLO checkpoint is optional while training is pending. Without it, types need review.
For a six-class Roboflow `.pt` trained at 640px, pass `--type-imgsz 640` to
`analyze-wall` or `segment-capture`. The inference notebook exposes the same size
as `TYPE_IMGSZ` and can preview the checkpoint on one photo with `RUN_YOLO_PREVIEW`.
Set `TYPE_MIN_SCORE` in the notebook segmentation cell or pass `--type-min-score`
to change the acceptance threshold for overlapping YOLO boxes. Changing this
threshold reuses completed SAM3 batches. `scene.json` reports unknown observations
with no overlapping candidate, a score below the cutoff, or an eligible box left
unmatched in the one-to-one assignment.
Low-confidence RGB proposals are retained separately. Joint detection/class scores are not a
six-class probability distribution. Undercling, sidepull and opening direction require additional
annotations; the six-class model cannot predict classes that are absent from its training labels.

## SLURM workflow

CPU stages use n2c48m24; GPU stages request c24m170-a100-2 / two A100s, all under
csci_ga_2271-2026fa. SAM3/COLMAP currently use GPU 0; YOLO training uses both GPUs through
torchrun. Scripts retain `--requeue` and persistent checkpoints.

```bash
export DEMO_PROJECT=/scratch/NetID/hold_detection
export DEMO_PYTHON=/scratch/NetID/runtime/bin/python
export DEMO_SOURCE=/scratch/NetID/captures/wall.mov
export DEMO_RUN=/scratch/NetID/hold-runs/wallC_video
export DEMO_MODE=3d
export DEMO_SAM_WEIGHTS=/scratch/NetID/models/sam3/sam3.pt
export DEMO_TYPE_WEIGHTS=/scratch/NetID/hold-runs/yolo26m/weights/best.pt
# Optional, e.g. 640 for a 640px Roboflow checkpoint.
# export DEMO_TYPE_IMGSZ=640
export DEMO_COLMAP=/path/to/cuda-colmap/bin/colmap

# Option A: one job executes all stages.
DEMO_STAGE=analyze sbatch scripts/slurm/wall-gpu.sbatch

# Option B: CPU/GPU jobs with afterok dependencies.
bash scripts/slurm/submit_wall.sh
```

Choose one submission method for each run. In 2D use DEMO_MODE=2d and one photo as DEMO_SOURCE;
the chain is prepare → segment → export. In 3D it is prepare → reconstruct → segment → export.
Resubmitting the same run reuses completed stages/frames. Changed input/models require a new run.
Interrupted frame folders are preserved as frame_batches/incomplete_* before retry.
COLMAP reuses its database/workspace; an interrupted SfM stage may need to run again.

Individual stages also work:

```bash
DEMO_STAGE=prepare sbatch scripts/slurm/wall-cpu.sbatch
DEMO_STAGE=reconstruct sbatch scripts/slurm/wall-gpu.sbatch
DEMO_STAGE=segment sbatch scripts/slurm/wall-gpu.sbatch
DEMO_STAGE=export sbatch scripts/slurm/wall-cpu.sbatch
```

Run them in order or use dependencies. Do not allow concurrent writers to the same DEMO_RUN.
For a network-enabled CPU node, `DEMO_STAGE=fetch-sam3 sbatch scripts/slurm/wall-cpu.sbatch`
downloads the gated checkpoint using HF_TOKEN from the environment. No credentials are printed.
USR1 requests a pause between SAM3 frames. Exit 75 means paused; optional DEMO_AUTO_REQUEUE=1
requeues that job. Otherwise resubmit the chain to reuse completed outputs.

## Metric scale and review fields

Without a known scale, position_3d is in arbitrary units. Metric positions/sizes/protrusion and
volumes remain null. Set DEMO_SCALE_M_PER_UNIT / --scale-m-per-unit if calibration provides a
factor. Alternatively supply DEMO_SCALE_REFERENCE / --scale-reference containing two pixels
on an **undistorted** image and their known physical distance:

```json
{"image":"frame_00000.jpg","point_a_px":[200,400],"point_b_px":[600,400],"distance_m":0.5}
```

Both pixels need valid depth. This calibrates length, not gravity or wall orientation.
Do not substitute original distorted-image coordinates. aabb_volume_m3 is an approximate
bounding-box volume, not actual hold volume. Grip area, true volume, edge depth and hold-facing
normal stay null. Local wall normals and protrusion are reviewable estimates from visible data.
Occluded lips/openings cannot be recovered as measured facts from an incomplete mesh.

Human edits are a JSON map keyed by stable hold ID:

```json
{"hold_a1b2c3d4e5f6":{"hold_type":"undercling","opening_direction":"down","is_start":true,"route_ids":["blue_v5"]}}
```

Set DEMO_ANNOTATIONS and rerun only export, or pass --annotations-path to holds export-wall.
Semantic edits do not overwrite geometry. Unknown types require an explicit downstream cost
policy or human confirmation for a reviewed demo.

## Annotated GLB matching

Mesh anchors in scene.json describe the nearest reconstructed vertex, not a semantic ground-truth
hold identity. For blue V5/yellow V3 use an explicit transform from reconstruction units to the
annotated GLB's meter frame. Named hold/volume objects are matched one-to-one within a distance
gate, with unmatched holds retained:

```bash
holds match-annotated-glb runs/video_demo/scene.json \
  --annotated-glb /path/to/wallC_blueV5_marked.glb \
  --transform /path/to/scene_to_annotated.json --output runs/video_demo/scene_matched.json
```

On HPC set DEMO_ANNOTATED_GLB and DEMO_ANNOTATION_TRANSFORM, then run
`DEMO_STAGE=match sbatch scripts/slurm/wall-cpu.sbatch`. Independent COLMAP/Polycam frames must
be aligned before this step. Identity is valid only when their coordinate frames already agree.

The new workflow is checked with synthetic images/video, calibrated depth examples, real mesh
serialization and an external COLMAP-process fixture. Real photo/video reconstruction quality
and a full SAM3/YOLO26 wall GPU run have not yet been validated on this development machine.
