"""Evaluate a crop classifier against explicit per-image class labels."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import TYPES
from .pipeline import collect_images


def classification_metrics(truth: list[int], predicted: list[int]) -> dict:
    if not truth or len(truth) != len(predicted):
        raise ValueError("Evaluation requires nonempty, equally sized truth and predictions")
    if any(not 0 <= i < len(TYPES) for i in truth + predicted):
        raise ValueError("Class IDs must be in [0, 5]")
    confusion = np.zeros((len(TYPES), len(TYPES)), dtype=int)
    for actual, guess in zip(truth, predicted):
        confusion[actual, guess] += 1
    per_class = {}
    for i, label in enumerate(TYPES):
        tp = int(confusion[i, i])
        support, predicted_count = int(confusion[i].sum()), int(confusion[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {
        "sample_count": len(truth), "correct_count": int(confusion.trace()),
        "accuracy": float(confusion.trace() / len(truth)),
        "macro_f1_all_six_classes": sum(m["f1"] for m in per_class.values()) / len(TYPES),
        "class_order": list(TYPES), "confusion_matrix_true_rows": confusion.tolist(),
        "per_class": per_class,
    }


def evaluate_crops(dataset: Path, output: Path, classifier, model_metadata=None) -> dict:
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    images = collect_images(dataset / "images")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Evaluation image stems must be unique")
    truth, crops = [], []
    for path in images:
        label = (dataset / "labels" / f"{path.stem}.txt").read_text().strip()
        if label not in {str(i) for i in range(len(TYPES))}:
            raise ValueError(f"Expected one class ID in [0, 5] for {path.name}, got {label!r}")
        truth.append(int(label))
        with Image.open(path) as image:
            crops.append(image.convert("RGB"))
    scores = classifier.classify_crops(crops)
    predictions = [TYPES.index(max(row, key=row.get)) for row in scores]
    report = {
        "schema_version": "1.0", "dataset": str(dataset.resolve()),
        "models": model_metadata or {}, **classification_metrics(truth, predictions),
        "limitation": "Metrics describe only the supplied crops; dataset independence is not verified.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
