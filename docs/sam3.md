# SAM 3: setup, run, and inspect results

SAM 3 is a separate text-prompted instance detector/segmenter. In this mode
YOLO does not supply the boxes. The default prompt is `climbing hold`; the crop
CNN optionally assigns grip types afterward. SAM's concept score is not a grip
type score, and neither score is calibrated accuracy.

For Google Colab, use the comparison
[DeMoSearch_SAM3_Colab.ipynb](../notebooks/DeMoSearch_SAM3_Colab.ipynb).
It checks out the public project repository and compares YOLO with SAM on the
same frames, with interactive figures and optional COCO ground-truth metrics.
See [notebook controls and evaluation protocol](colab.md). Its GPU setup and
inference need to run in Colab; local tests are not a GPU readiness claim.

## What was actually verified here

- Official source at `7f473cfd983d5eb43040ff608f66bff0d7bdcb53` downloaded and installed.
- A separate locked environment exists at `runtimes/sam3/.venv`; base YOLO dependencies were preserved.
- Official `build_sam3_image_model` and `Sam3Processor` imports succeeded.
- Actual construction of the official positional encoder failed on CPU:
  `AssertionError: Torch not compiled with CUDA enabled`. The pinned source
  precomputes tensors on `device="cuda"`, even if the builder is passed `cpu`.
- Official Hugging Face checkpoint access failed at the cloud proxy with HTTP 403,
  including an escalated connectivity check. No SAM 3 checkpoint was downloaded.
- Tests verify SAM's output conversion, NMS/mask alignment, classification-score
  separation, artifact export and gallery generation with controlled model outputs.
  This does not validate the learned model's predictions.
- Real YOLO inference was rerun after the changes and produces the same 51
  candidates for the first frame at confidence 0.35. The 38-frame run has an HTML gallery.

**No actual SAM 3 segmentation on gym photos has been verified yet.** The current
machine has no NVIDIA GPU. Use a CUDA machine for the steps below.

## 1. Install on a Linux/WSL2 machine with an NVIDIA GPU

Use Python 3.12 and a driver compatible with CUDA 12.6. From the project root:

```bash
uv sync --project runtimes/sam3 --frozen
uv pip install --python runtimes/sam3/.venv/bin/python --reinstall \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
runtimes/sam3/.venv/bin/holds sam3-doctor
```

The locked install is CPU-compatible so it can also install tooling on this
cloud instance. The second command replaces PyTorch with official CUDA wheels.
After this command, use the environment's `holds` binary directly: another
`uv sync` or plain `uv run` restores its CPU lock. `cuda_available` in the doctor
output must be true before inference. Native macOS has no NVIDIA CUDA support;
use a Linux GPU machine, a GPU notebook or a university GPU node instead.

The official model has 848M parameters. Start with one photo; memory and speed
on a GPU have not been measured in this project.

## 2. Obtain the official weights

Request/accept access at <https://huggingface.co/facebook/sam3>. After approval,
authenticate on your own machine:

```bash
runtimes/sam3/.venv/bin/hf auth login
runtimes/sam3/.venv/bin/holds fetch-sam3
runtimes/sam3/.venv/bin/holds fetch-models --model classifier
```

The login command prompts securely; do not place a token in a command argument,
source file or chat. The downloader resolves `main` to an exact Hub commit,
checks the file against the official SHA-256 ETag when available, and records
the resolved revision and SHA-256 in `models/sam3/checkpoint.json`.

For a checkpoint already downloaded from the official repository, pass its local
path with `--weights /path/to/sam3.pt`. Runtime inference with a local checkpoint
uses `load_from_HF=False` and does not need a Hugging Face network request.

For **this cloud environment**, a draft was saved with custom allowed domains
`huggingface.co` and `cas-bridge.xethub.hf.co`, plus a secure `HF_TOKEN` requirement
scoped to `huggingface.co`. Preset destinations were preserved. This saved draft
does not apply networking, supply the token, add a GPU, or publish a snapshot.
Review and save it in environment settings, enter a read token for an account
with approved model access, then publish the environment as required by the UI.
The original network blocker remains until the relevant settings take effect.

## 3. Run on your photos

For a JPEG/PNG file or directory of RGB photos:

```bash
runtimes/sam3/.venv/bin/holds sam3 path/to/photos \
  --output runs/sam3-test --limit 1 --prompt 'climbing hold'
```

For the existing Polycam frames, which need a clockwise rotation for display:

```bash
runtimes/sam3/.venv/bin/holds sam3 data/wallA_1/keyframes/corrected_images \
  --output runs/sam3-wallA --limit 1 --rotate 90 --conf 0.5
```

Remove `--limit 1` after inspecting the first result. Lowering `--conf` can
increase recall and false positives. Compare prompts such as `climbing hold`
and `rock climbing hold` in separate output directories. To inspect pure SAM
segmentation without the crop classifier:

```bash
runtimes/sam3/.venv/bin/holds sam3 path/to/wall.jpg \
  --output runs/sam3-masks --classifier none
```

Without the classifier, grip type and type score remain unknown/null. SAM's
instance confidence must not be interpreted as confidence in jug/crimp/etc.
Changing the prompt changes detection, not the classifier's class vocabulary.

## 4. View results

Open **`<output>/index.html` in a browser**. No server or API service is needed.
Each frame shows the annotated photo; expand its details for individual crops,
type scores, review flags and links to binary mask PNGs. You can filter the crops
to candidates needing review. The gallery uses relative paths: copy/download the
whole output folder rather than its HTML file alone.

For an old run created before galleries were added:

```bash
.venv/bin/holds gallery runs/wallA_1-detector
```

That particular gallery contains real **YOLO** outputs, not SAM masks. The file
`holds.json` in each frame contains boxes, processed-image coordinates, mask
paths and scores. The summary records the model hash, prompt and source revision.

The Colab notebook fetches its own project checkout from public GitHub.
For a GPU machine, use the repository checkout or
the prepared source archive and follow the installation commands above.
