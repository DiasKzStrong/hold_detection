# Wall perception

| Module under `src/hold_detection/` | Responsibility |
| --- | --- |
| `capture/` | Prepare one photo, a photo set, or video. Apply EXIF orientation and optional 2D crop. |
| `perception/` | Run SAM 3, assign YOLO type proposals, and measure observed hold colors. |
| `reconstruction/` | Run CUDA COLMAP, read camera poses and depth, and resolve scale and up direction. |
| `scene/` | Lift masks into 3D, fuse observations, export `scene.json`, and optionally match annotated GLB IDs. |
| `review/` | Render masks, mesh markers, offline galleries and the Colab up picker. |
| `training/` | Prepare labeled data and train or evaluate YOLO26 and the crop-classifier experiments. |
| `legacy/` | Draft Polycam RGB-D fusion and registration for the comparison workflow. |
| `cli.py`, `wall_commands.py`, `wall_pipeline.py` | Expose CLI stages and run the complete analysis. |

Start with [DeMoSearch_Pipeline.ipynb](../notebooks/DeMoSearch_Pipeline.ipynb)
or [open it in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Pipeline.ipynb).
Its cells call the same CLI stages as SLURM. Edit the notebook directly and provide
existing SAM 3 and YOLO checkpoints. The comparison and crop-classifier notebooks live
in [notebooks/experiments](../notebooks/experiments). Train YOLO through
`scripts/train_yolo26.py` or the [YOLO SLURM jobs](yolo26-training.md).

Python callers can import `prepare_capture` from `hold_detection.capture`,
`reconstruct_wall` from `hold_detection.reconstruction`, `segment_capture` from
`hold_detection.perception.segmentation`, and `export_wall_scene` from
`hold_detection.scene`. `hold_detection.wall_pipeline.analyze_wall` runs the full workflow.

The inference notebook has separate `INPUT_PATH_2D` and `INPUT_PATH_3D` settings.
The default 3D preset stages 15 overlapping RGB keyframes of the Blue V5 section
from `wallC_raw_data.zip`. For a new wall section, point `INPUT_PATH_3D` to its own
photo folder, ZIP, or video. The notebook can show saved masks next to each photo,
color them by type or observed RGB, and display the generated GLB with hold markers.
The same review ID identifies a hold on the mesh and in its supporting frames.

## Input and output

2D accepts exactly one photo (including iPhone HEIC) and exports masks, IDs, normalized 2D centers/sizes, observed colors
and optional six-class RGB type proposals. 3D accepts ordinary overlapping photos or a video,
constructs a local wall through COLMAP, segments registered **undistorted images** and lifts masks
using their stereo depth maps. Repeated evidence is fused through color consistency and 3D
sample overlap. Single-view and unmapped holds remain in the output for review.

Enable `USE_2D_CROP` in the inference notebook to select the wall section using a
rectangle and preview the crop. The `prepare-capture --crop-xyxy` CLI option accepts
normalized left, top, right, and bottom edges on the EXIF-oriented source.
The 2D coordinates refer to the prepared crop and `wall.input_crop` records its
pixel bounds in the oriented source.

Both modes return one schema 5.0 `RUN/scene.json` with the same hold keys. The
top-level `units`, `up_axis` and `source_file` describe the frame of `holds[].position`.
The export follows the Blue V5 field names for the planner while retaining image
observations and mesh correspondence for review.

| Field | 2D photo | 3D wall |
| --- | --- | --- |
| `position` | Normalized `[x, y]` in the prepared photo, with right and down positive | `[x, y, z]` in the exported GLB, or `null` for an unmapped hold |
| `units` | `normalized_image` | `meters` with scale calibration, otherwise `arbitrary` |
| `up_axis` | `null` | `y` when orientation is resolved, otherwise `null` |
| `height_m` | `null` | Height from the scanned wall bottom with orientation and metric scale |
| `mean_color` | Mean RGB of masked image pixels | Average of per-view RGB means for merged masks |
| `is_start`, `is_top` | Explicit annotations | Explicit annotations |

The route-grouping consumer clusters holds using `holds[].id` and `holds[].mean_color`. `mean_color` contains RGB channels in the range 0–255 or `null` when unavailable. Each `observations[]` entry provides per-view color evidence, including `mean_color_source` and an optional `color_lab`. Manual route annotations populate `route_ids`.

Color can fall back to the bounding box when a mask is unavailable. The hold's
`flags` record this fallback. `start_hold_ids` lists the marked starts, and
`top_hold_id` is populated when exactly one hold is marked as the finish.

The 3D scene exporter estimates `wall.up_vector` from registered camera poses and
records the result in `wall.up_orientation`. Upright views with enough variation
in camera direction can provide a usable estimate. Ambiguous captures return
`manual_required`.

The Colab picker accepts two clicks on a known vertical line in an undistorted
registered image. The CLI accepts the same reference with `export-wall --up-reference`.
An example reference file is
`{"image":"frame_00003.jpg","bottom_px":[320,550],"top_px":[320,180]}`.
Both points need valid, locally consistent stereo depth. This export uses
the existing reconstruction and inference files.

`height_from_wall_bottom` is the hold's GLB Y minus `wall_bottom_height`, the
0.5th percentile of GLB vertex heights. The reference describes the scanned area.
In a crop of the middle of a wall, zero is near the lower edge of that crop.

`height_m` uses the same reference with metric scale. `height_3d` gives the
projection relative to the reconstruction origin.

When the up direction is resolved, scene export writes `wall_upright.glb` with
glTF +Y pointing up. `wall.glb_path` points to this file, and `wall.reconstruction_to_glb`
records the 4×4 transform from the reconstruction, including any metric scale.
Use `position` or `position_glb` to place holds in the GLB. The hold-level
`wall_normal` shares those axes. `wall_normal_reconstruction` gives the same
vector in the camera reconstruction's coordinate frame.

```text
RUN/capture/capture.json
RUN/reconstruction/reconstruction.json   # 3D only
RUN/reconstruction/wall.glb              # raw reconstruction checkpoint
RUN/wall_upright.glb                     # oriented export when up is resolved
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
Set `TYPE_MIN_SCORE` in the notebook settings or pass `--type-min-score`
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
Occluded lips and openings leave grip fields null. `footprint_m` estimates across-wall
and up-wall spans by projecting visible hold samples onto the local wall plane.
`wall_angle_deg` comes from the camera-facing wall normal and up vector. Negative
angles indicate overhangs and positive angles indicate slabs outside the vertical tolerance.

Human edits are a JSON map keyed by stable hold ID:

```json
{
  "hold_a1b2c3d4e5f6": {"hold_type":"undercling","opening_direction":"down","is_start":true,"route_ids":["blue_v5"]},
  "hold_112233445566": {"is_top":true,"route_ids":["blue_v5"]}
}
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
