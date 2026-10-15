"""Image-space boxes; no metric geometry is inferred from RGB."""

from dataclasses import dataclass
import math

TYPES = ("crimp", "jug", "pinch", "pocket", "sloper", "volume")


def canonical_type(label: str) -> str:
    label = label.strip().lower()
    label = {"slope": "sloper", "juges": "jug"}.get(label, label)
    if label not in TYPES:
        raise ValueError(f"Unsupported hold class {label!r}; expected {TYPES}")
    return label


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    label: str
    label_is_type: bool = True


def box_iou(a, b) -> float:
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def nms_indices(detections: list[Detection], iou_threshold: float) -> list[int]:
    """Return stable indices so associated masks stay aligned with boxes."""
    remaining = sorted(range(len(detections)), key=lambda i: detections[i].score, reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [i for i in remaining
                     if box_iou(detections[best].bbox, detections[i].bbox) <= iou_threshold]
    return sorted(kept, key=lambda i: (detections[i].bbox[1], detections[i].bbox[0]))


def merge_detections(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Class-agnostic NMS: two type guesses must not become two physical holds."""
    return [detections[i] for i in nms_indices(detections, iou_threshold)]


def tile_windows(width: int, height: int, size: int, overlap: float):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Tile size must be positive and overlap must be in [0, 1)")
    step = max(1, round(size * (1 - overlap)))

    def origins(length):
        last = max(0, length - size)
        return sorted(set(range(0, last + 1, step)) | {last})

    return [
        (x, y, min(x + size, width), min(y + size, height))
        for y in origins(height) for x in origins(width)
    ]


def crop_box(bbox, width: int, height: int, padding: float = 0.1):
    x1, y1, x2, y2 = bbox
    pad = max(x2 - x1, y2 - y1) * padding
    return (
        max(0, math.floor(x1 - pad)), max(0, math.floor(y1 - pad)),
        min(width, math.ceil(x2 + pad)), min(height, math.ceil(y2 + pad)),
    )
