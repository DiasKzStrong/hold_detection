# Reference checkpoints

The user selected [Marcuss2106/indoor-climbing-gym-hold-classification](https://github.com/Marcuss2106/indoor-climbing-gym-hold-classification).
Both checkpoints are pinned to commit `06a763cd8480a8c5cf2b22ec47c749b8bf4fe97d`.
The downloader verifies these SHA-256 digests before installing the files:

| Artifact | Upstream path | SHA-256 |
| --- | --- | --- |
| Six-class YOLO11s detector | `runs/detect/yolov11s-climbing-gym/weights/best.pt` | `5da8ce4bf43c459c618ae0333e136b88759035d2aa43e45ef7a96a4647e15f72` |
| Two-convolution crop CNN, epoch 24 | `sample_checkpoint.pth` | `19ae7c675b2846dddeef8a5eadb0b7ede21cd8023233db2fabeadc6c8bb6aa0a` |

Class order: crimp, jug, pinch, pocket, slope, volume. This project normalizes
`slope` to `sloper` and separates `object_kind=volume` from grip types.
The CNN adapter matches the upstream architecture, normalization and 128-pixel
aspect-preserving crop preprocessing. Checkpoints stay in ignored `models/`;
they are not redistributed as repository source files.

Upstream links the Diego Spaziani climbing-hold dataset:
<https://doi.org/10.34740/KAGGLE/DSV/13795579>. Its README describes both real
and synthetic images. We have not audited the dataset or its split, downloaded
it, retrained the checkpoints, or independently reproduced its reported metrics.

No LICENSE file was found in the selected reference revision. Its code and
weight licensing must not be assumed to be MIT or otherwise unrestricted.
Confirm permission/terms before redistribution or use outside the course.
Ultralytics and any optional SAM dependencies also have their own license terms.

## Known behavior on our Polycam frames

The YOLO model finds many holds, but also detects some route tags as holds and
can produce overly broad boxes. The CNN sometimes strongly disagrees with the
detector and may be confidently wrong. Its scores are not calibrated. CNN
disagreements and low type scores are exported for human review; they are not
silently treated as ground truth. Neither model includes a `non_hold` class.

Two green V3 routes are described in the provided notes. Color alone cannot
determine route membership. Detection counts across frames count repeated
observations; they are not counts of unique holds on the wall.

## Optional segmentation

The SAM adapter targets official SAM 2.1, with a tiny checkpoint by default:
<https://github.com/facebookresearch/sam2>.
Its mask-quality score is separate from the detector and classifier scores.
Without SAM, `mask_path` and `segmentation_method` are null: a bounding box is
not reported as a segmentation mask.
