# YOLO vs SAM 3 · Colab experiment

[Open in Colab](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_SAM3_Colab.ipynb).
Select a Python 3.12 GPU runtime and execute sections 1–8 in order. The default
is three frames, not the whole scan. YOLO also works on CPU without HF access.

## What you see

1. Collapsed setup cells and a contact sheet to check image rotation.
2. A real YOLO11s baseline before any SAM setup or Hugging Face authorization.
3. Independent SAM 3 text-prompted instances followed by crop-CNN grip types.
4. Original / YOLO / SAM side by side; a frame selector, per-model score sliders,
   boxes/masks/ID toggles, and a separate crop grid with readable type scores.
5. Small metric tables and plots, plus optional TP/FP/FN error overlays.
6. One downloadable experiment ZIP with per-model galleries, masks, crops,
   comparison images, `metrics.csv`, `comparison.json`, and `session.json`.

If SAM access or GPU setup fails, the notebook retains YOLO results and explicitly
marks SAM as not run. It never substitutes boxes or simulated masks for SAM output.

## Runtime and inputs

The setup fetches this public GitHub repository at `CODE_REF` (default `main`),
checks out the resolved commit, and records its SHA. It refuses to overwrite
local checkout changes. No base64 source payload is embedded in the notebook.
For reproduction, set `CODE_REF` to a recorded commit instead of `main`.

An isolated system-site-packages venv reuses Colab's CUDA PyTorch/torchvision/Triton
without replacing them with this repository's CPU wheels. NumPy 1.26 and model
dependencies live in that venv; tables/widgets use the notebook kernel. Installation
and inference logs are saved under `/content/demosearch-comparison/logs` instead
of filling the notebook. The official SAM revision is
`7f473cfd983d5eb43040ff608f66bff0d7bdcb53`. Native BF16 is used where supported,
otherwise FP16, including on T4.

SAM weights require approved access at <https://huggingface.co/facebook/sam3>.
Provide a read token through the Colab Secret `HF_TOKEN` or the hidden prompt.
Tokens and weights are outside the exported experiment directory.

Upload JPEG/PNG images, a photo ZIP, a Polycam raw-data ZIP, or select a Drive
folder/archive. Mesh textures, depth-only archives and unsafe ZIP paths are
rejected. Polycam exports retain metadata, but only RGB images go to inference.
`ROTATE=-1` means 90° clockwise for our Polycam scan and 0° otherwise; always
check the preview. `MAX_IMAGES=0` processes all frames. New experiment/run/report
names preserve earlier results when cells are rerun.

## Honest metrics

Without ground truth, only latency, candidate counts and score distributions
are measured. Scores are not accuracy, nor calibrated across YOLO and SAM.
Counts are observations across frames, not distinct physical holds.

For quality metrics, enable `USE_GROUND_TRUTH` and upload one COCO instance JSON:

- Annotate **all** visible holds/volumes on each evaluated frame, not only some
  route colors. Every prediction frame must have a uniquely matching COCO image
  filename and correct dimensions. Extra GT frames are excluded and counted.
- Category `hold` gives class-agnostic localization metrics. Categories
  `crimp`, `jug`, `pinch`, `pocket`, `sloper`, `volume` enable type metrics too.
- Polygon or COCO RLE masks are needed for mask AP; masks must exist for every GT
  instance. Box-only YOLO has mask AP **N/A**, not zero. Crowd regions are rejected;
  annotate separate instances. Do not evaluate an unlabeled template as negative GT.
- `GT_COORDINATES=source` rotates original-image labels to the processed frame;
  use `processed` only for labels drawn on already-rotated images.

P/R/F1 use the current viewer thresholds at the moment the metrics cell runs.
Rerun it after changing sliders. Matching is confidence-ordered, one-to-one,
class-agnostic, at box IoU ≥ 0.5. TP/FP/FN error overlays use that same snapshot.

Box/mask AP use pycocotools, the COCO 0.50:0.05:0.95 IoU grid and **all saved
proposals**, independent of viewer thresholds. Default candidate floor is 0.05;
AP is truncated by that floor and by upstream NMS. **maxDets=500**, intentionally
different from standard COCO's 100, is recorded in JSON. Mask IoU is conditional
on box-matched pairs and therefore does not by itself penalize missed objects;
use mask AP alongside it.

Type accuracy, six-class macro-F1 and confusion matrices use raw top-1 types
on IoU-matched typed GT, not the thresholded `unknown` display label. Separately,
type coverage and end-to-end correct-type recall over **all typed GT** expose
missed holds. Generic `hold` labels cannot establish grip classification quality.
YOLO's class head and SAM's separate CNN are different classification baselines;
localization metrics deliberately ignore type.

Model latency excludes weight loading, includes the first-frame warmup, and
synchronizes CUDA. It includes detection/segmentation and classification, not
crop/PNG export. p50/p95 on three frames are smoke-test diagnostics, not a stable
throughput benchmark. Runtime/GPU/package versions are exported with results.

## Download and verification

Download the ZIP before ending the Colab session. Extract the entire archive and
open `report_…/index.html`; both full per-model galleries are also inside.
Static figures work offline; live widgets require the active notebook and inputs.

The notebook passes nbformat validation, code-cell parsing and deterministic
regeneration tests. Evaluation tests cover true/false positives, missing objects,
rotation, polygon masks, confidence filtering, and actual pycocotools AP calculations.
Real YOLO inference and the new static visualization were checked on three user
Polycam frames. These unlabeled frames provide no accuracy estimate.

**Actual SAM GPU/Colab inference has not been executed on the CPU-only development
machine.** Structural tests and synthetic GT tests do not establish model accuracy
or GPU memory requirements. The crop CNN's earlier 69/100 sample result is not
performance on this gym. Split future evaluations by physical wall/scan, not by
random neighboring frames.

Regenerate and test:

```bash
python scripts/build_colab_notebook.py
uv run --extra dev pytest
```
