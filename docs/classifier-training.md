# Supervised crop classifiers: ResNet18 and SigLIP

[Open the training notebook](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Classifier_Training.ipynb).
For full training with the course allocation, see [NYU Cloud Bursting / SLURM](nyu-hpc.md).

The models solve **crop classification**, not hold detection:

```text
Labeled YOLO/COCO export → GT box crops → train/validation/test manifests
                                            ├─ ImageNet ResNet18 fine-tuning
                                            └─ frozen SigLIP → cached features → linear head

Wall photographs → SAM 3 boxes/masks → selected trained classifier → six hold types
```

## Data requirements and provenance

The linked [Roboflow project](https://universe.roboflow.com/climbing-holds-classification/holdclassificatore_v3_mix)
publishes 2,282 images, six classes and box annotations. Published source splits
are 1,854 train / 215 validation / 213 test; class counts sum to about 62,289 boxes.
It is not an instance-mask dataset. Download a **YOLO detection** or **COCO detection**
export, not a classification export. Keep its version, attribution and license files.
Individual unannotated Polycam photographs cannot train a supervised classifier.

Our upstream YOLO/CNN cite the Diego Spaziani Kaggle dataset. Roboflow version 2
is named “Kaggle Version”; source photographs were found in both that project
and Kaggle's training directory, including `20250514_223854` and `20250522_160413`.
The reviewed variants differ in resolution, grayscale and/or mirroring. This
confirms shared sources, not byte-for-byte identity of every export or split.
Do not treat this dataset as an independent new test of the upstream checkpoint.
Roboflow lists CC BY 4.0; the Kaggle listing specifies CC BY-SA 4.0. Retain the
actual downloaded version's terms and attribution; do not assume identical licenses.

## 1. Install and prepare crops (CPU)

Local CPU development:

```bash
uv sync --frozen --extra training --extra dev
.venv/bin/holds prepare-crops /path/to/labeled-export.zip --output data/hold-crops
```

For CUDA/HPC, install matching CUDA wheels first, then `pip install -e '.[training]'`.
Do not run the root `uv sync` afterward: its default Linux lock uses CPU wheels.

Supported layouts:

```text
data.yaml                    train/_annotations.coco.json
train/images/*.jpg           train/*.jpg
train/labels/*.txt           valid/_annotations.coco.json
valid/images/*.jpg           valid/*.jpg
valid/labels/*.txt
test/images, test/labels     test/ is optional in both layouts
```

`images/train` + `labels/train` is also supported. Nested ZIPs with one export
root, and Kaggle's `Final_Dataset`, are recognized. A YOLO export needs `data.yaml`
with six names. If absent, explicitly provide names **in annotation class-ID order**:

```bash
holds prepare-crops /path/to/Final_Dataset --output data/hold-crops \
  --class-names Crimp Jug Pinch Pocket Slope Volume
```

`Slope` is normalized to `sloper`. COCO category IDs may be noncontiguous.
Boxes, references, dimensions, ZIP traversal/symlinks, duplicate paths, and EXIF
orientation are checked. YOLO polygons are intentionally rejected. Crops use
10% padding by default, JPEG quality 95, and a maximum side of 512 pixels.

Preparation never overwrites an output directory. It writes `manifest.json`,
source-image identities/pixel hashes, class counts and crop files. ZIP source
files are retained under `source/` for audit; do not include datasets in Git.

### Split safety

All crops retain their **source image's split**. Exact identical pixels crossing
splits are rejected by default. `--allow-split-overlap` is an explicit, recorded
override, not an independence guarantee. Suspected `.rf.` filename families are
reported separately because common names such as `image_0` can be ambiguous.

Pixel hashes do not catch resized/mirrored copies or nearby views of one wall.
Audit the source dataset; do not randomly split the resulting crops. For a strict
wall/scan audit, supply a complete CSV relative to the selected export root:

```csv
source_image,group_id
train/images/wall_A_01.jpg,wall_A
train/images/wall_A_02.jpg,wall_A
valid/images/wall_B_01.jpg,wall_B
```

Pass `--groups groups.csv`. A group crossing splits is rejected; this command
does not silently repartition your source dataset. Fix its split assignment first.

## 2. Fine-tune ResNet18

```bash
holds train-classifier --architecture resnet18 --dataset data/hold-crops \
  --output runs/resnet18 --epochs 20 --batch-size 128 --workers 8 --device cuda:0
```

The default initializes torchvision ImageNet1K V1 weights and fine-tunes the full
encoder plus a six-class head with AdamW and cosine LR decay (default LR 1e-4).
Inputs use aspect-preserving 224-pixel letterboxing and ImageNet normalization.
Training adds horizontal flips and mild color jitter; validation/test are deterministic.
Class-weighted cross entropy uses **training counts only**. The exported model
records preprocessing, class order, initialization and crop padding.

## 3. Frozen SigLIP + linear head

Extract features once on GPU:

```bash
holds extract-features --dataset data/hold-crops --output runs/siglip-features \
  --batch-size 128 --workers 8 --device cuda:0
```

`google/siglip-base-patch16-224` uses its vision encoder and pooling output, not
zero-shot text prompts. The Hub revision resolves to an exact commit, and the
cached encoder is SHA-256 checked. Inputs use recorded SigLIP normalization,
aspect-preserving letterboxing and L2-normalized 768-dimensional embeddings.
The frozen encoder runs in eval mode without gradient computation or random crop
augmentation. CPU/GPU extraction is currently FP32, not an AMP throughput claim.

Feature batches are atomically saved under train/val/test. Rerun the same command
after interruption: completed shards are reused without decoding their source
images again. Keep batch size, dataset and model configuration unchanged.
At approximately 62k objects, float32 embeddings alone are about 183 MiB; the
encoder, metadata, checkpoints and source crops require additional disk space.

Train the linear head on CPU without loading/running the encoder:

```bash
holds train-classifier --architecture siglip --features runs/siglip-features \
  --output runs/siglip-head --epochs 30 --batch-size 512 --workers 0 --device cpu
```

The default head LR is 0.01. No encoder fine-tuning or text tower is used. This
is a supervised linear probe. `best.pt` packages the frozen encoder once at the
end so inference does not need Hub access. It is larger than the linear head alone.

## Checkpoint/restart and metrics

`last.pt` contains model/head and optimizer/scheduler states, epoch, next batch,
partial-epoch loss and RNG state. Checkpoints are saved every 100 batches by
default, at epoch boundaries, and at a safe batch boundary after SIGUSR1/SIGTERM
or `--time-budget-seconds`. Stateless per-image augmentation supports batch-offset
restart. Bitwise reproducibility across different GPU types/counts is not promised.

Rerun training with `--resume` and the same output, epochs, batch size, LR, seed,
initialization and data. Operational time budget and worker count may change.
Exit code **75** means a checkpointed pause. Do not treat it as completed training.
SigLIP extraction resumes via feature shards rather than `last.pt`.

Best epochs are selected by **validation all-six-class macro-F1**. The test split
is evaluated after training, never used for epoch selection. Outputs include:

- `best.pt`: portable inference artifact; load with `weights_only=True`.
- `last.pt`: restart state; SigLIP also has a small `best_head.pt`.
- `history.csv`, learning curves, validation/test confusion matrices.
- `metrics.json` with per-class P/R/F1, accuracy, macro-F1 and sample counts.
- `val_predictions.jsonl` / `test_predictions.jsonl` with per-crop scores and truth.

Training loss is class-weighted; reported validation/test loss is unweighted.
Softmax scores are not calibrated probabilities of correctness. Missing validation
classes contribute zero to all-six-class macro-F1 and have explicit support counts.

## 4. Inference on your photos

```bash
holds sam3 /path/to/photos --output runs/sam3-siglip --classifier trained \
  --classifier-weights runs/siglip-head/best.pt

holds predict /path/to/photos --output runs/yolo-resnet --classifier trained \
  --classifier-weights runs/resnet18/best.pt
```

The SAM/YOLO comparison notebook has `CLASSIFIER='trained'` and
`CLASSIFIER_WEIGHTS='/path/to/best.pt'` settings. For SigLIP the runtime needs the
training extra; pretrained weights are already inside the artifact. SAM still
requires its separate official checkpoint, GPU and approved access when downloading.

Train crops use GT boxes; runtime crops use imperfect detector boxes. Evaluate
this shift on separately annotated gym photos. Neither new classifier has a
trained background/non-hold class. Low confidence becomes `unknown`; confident
background mistakes remain possible. Masks are exported, but classifier inputs
are RGB box crops, not masked foreground. 3D/depth are not used.

## Verification status

Automated tests exercise YOLO/COCO preparation, split leakage checks, actual
ResNet training/checkpoint restart, the real SigLIP API with a tiny random offline
test configuration, feature-cache reuse, portable inference and SLURM syntax.
Real pretrained ImageNet ResNet18 and public SigLIP weights were also loaded and
forward-tested on CPU. Tiny/random test models are explicitly marked test-only.
No accuracy from these fixtures represents hold recognition performance.

**Full-dataset training and two-A100 execution have not been run from this workspace.**
NYU jobs must be submitted from your authorized Cloud Bursting environment.
