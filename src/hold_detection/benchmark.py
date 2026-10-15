"""Shared-image diagnostics and explicit COCO-ground-truth evaluation.

Quality metrics are never inferred from model scores or from agreement between models.
"""

from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .geometry import TYPES, box_iou, canonical_type


def load_run(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    files = sorted(path.glob("*/holds.json"))
    if len(files) != summary["frame_count"]:
        raise ValueError(f"Incomplete run: {path}")
    frames = []
    for file in files:
        frame = json.loads(file.read_text())
        frame["directory"] = str(file.parent.resolve())
        frames.append(frame)
    if len({f["source_image"] for f in frames}) != len(frames):
        raise ValueError("A run must contain at most one prediction per source image")
    return {"path": str(path.resolve()), "summary": summary, "frames": frames}


def validate_shared_frames(runs: dict) -> None:
    reference = next(iter(runs.values()))["frames"]
    expected = {f["source_image"]: (f["source_size_wh"], f["processed_size_wh"],
                                   f["rotation_clockwise_degrees"]) for f in reference}
    for name, run in runs.items():
        actual = {f["source_image"]: (f["source_size_wh"], f["processed_size_wh"],
                                     f["rotation_clockwise_degrees"]) for f in run["frames"]}
        if actual != expected:
            raise ValueError(f"{name}: comparison requires identical source images, sizes and rotation")


def detection_matches(predictions: list[dict], truth: list[dict], threshold=0.5) -> dict:
    """Confidence-ordered, one-to-one, class-agnostic matching at a fixed IoU."""
    available = set(range(len(truth)))
    pairs, false_positives = [], []
    for p in sorted(range(len(predictions)), key=lambda i: predictions[i]["detector_score"], reverse=True):
        overlaps = [(box_iou(predictions[p]["bbox_xyxy"], truth[g]["bbox_xyxy"]), g)
                    for g in sorted(available)]
        overlap, g = max(overlaps, default=(0.0, None), key=lambda item: item[0])
        if g is not None and overlap >= threshold:
            available.remove(g)
            pairs.append({"prediction": p, "truth": g, "iou": overlap})
        else:
            false_positives.append(p)
    return {"pairs": pairs, "fp": false_positives, "fn": sorted(available)}


def rotate_box(box: list[float], width: int, height: int, clockwise: int) -> list[float]:
    x1, y1, x2, y2 = box
    return {0: [x1, y1, x2, y2], 90: [height - y2, x1, height - y1, x2],
            180: [width - x2, height - y2, width - x1, height - y1],
            270: [y1, width - x2, y2, width - x1]}[clockwise]


def _mask_from_coco(segmentation, width, height):
    from pycocotools import mask as mask_utils
    if not segmentation:
        return None
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle).astype(bool)
    if mask.ndim == 3:
        mask = mask.any(axis=2)
    if mask.shape != (height, width) or not mask.any():
        raise ValueError("Invalid/empty ground-truth segmentation")
    return mask


def load_ground_truth(path: Path, frames: list[dict], coordinates="source") -> dict:
    if coordinates not in {"source", "processed"}:
        raise ValueError("Ground-truth coordinates must be source or processed")
    data = json.loads(path.read_text())
    if data.get("annotation_status") == "unlabeled":
        raise ValueError("This is an unlabeled template; annotate it before evaluating")
    images = data.get("images", [])
    categories = data.get("categories", [])
    if not images or not categories:
        raise ValueError("Expected COCO images, categories and annotations")
    if len({i["id"] for i in images}) != len(images):
        raise ValueError("Duplicate COCO image IDs")
    category_types = {}
    for category in categories:
        label = category["name"].strip().lower()
        if label in {"hold", "holds", "climbing hold", "climbing_hold", "foothold"}:
            category_types[category["id"]] = None
        else:
            category_types[category["id"]] = canonical_type(label)
    if len(category_types) != len(categories):
        raise ValueError("Duplicate COCO category IDs")
    annotation_ids = [a["id"] for a in data.get("annotations", [])]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError("Duplicate COCO annotation IDs")
    grouped = {i["id"]: [] for i in images}
    for annotation in data.get("annotations", []):
        if annotation.get("iscrowd", 0):
            raise ValueError("Crowd annotations are unsupported; annotate each hold separately")
        if annotation["image_id"] not in grouped or annotation["category_id"] not in category_types:
            raise ValueError("Annotation references an unknown image/category")
        grouped[annotation["image_id"]].append(annotation)
    result = {}
    used = set()
    for frame in frames:
        source = frame["source_image"].replace("\\", "/")
        exact = [i for i in images if source == i["file_name"] or source.endswith("/" + i["file_name"])]
        candidates = exact or [i for i in images if PurePosixPath(source).name == PurePosixPath(i["file_name"]).name]
        if len(candidates) != 1:
            raise ValueError(f"Ground truth must match {source} uniquely; found {len(candidates)} images")
        image = candidates[0]
        if image["id"] in used:
            raise ValueError("Two prediction images map to the same ground-truth image")
        used.add(image["id"])
        width, height = frame["source_size_wh"] if coordinates == "source" else frame["processed_size_wh"]
        if (image["width"], image["height"]) != (width, height):
            raise ValueError(f"{image['file_name']}: GT dimensions do not match {coordinates} coordinates")
        rotation = frame["rotation_clockwise_degrees"] if coordinates == "source" else 0
        truth = []
        for annotation in grouped[image["id"]]:
            x, y, w, h = annotation["bbox"]
            if (not all(math.isfinite(v) for v in [x, y, w, h]) or w <= 0 or h <= 0
                or x < 0 or y < 0 or x + w > width + 1e-4 or y + h > height + 1e-4):
                raise ValueError(f"Invalid/out-of-bounds COCO box: annotation {annotation['id']}")
            mask = _mask_from_coco(annotation.get("segmentation"), width, height)
            if mask is not None and rotation:
                mask = np.rot90(mask, k=-(rotation // 90))
            truth.append({"id": annotation["id"],
                          "bbox_xyxy": rotate_box([x, y, x + w, y + h], width, height, rotation),
                          "type": category_types[annotation["category_id"]], "mask": mask})
        result[source] = truth
    return {"frames": result, "selected_images": len(result),
            "unused_gt_images": len(images) - len(used), "coordinates": coordinates}


def _read_mask(frame, hold):
    name = hold.get("mask_path")
    if not name:
        return None
    root = Path(frame["directory"]).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Mask path escapes the frame directory")
    with Image.open(path) as image:
        mask = np.asarray(image) > 0
    if mask.shape != tuple(reversed(frame["processed_size_wh"])):
        raise ValueError("Prediction mask dimensions differ from the image")
    return mask


def _coco_ap(frames, truth, segmentation=False, max_detections=500):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as mask_utils
    dataset = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "hold_or_volume"}]}
    predictions, annotation_id = [], 1
    for image_id, frame in enumerate(frames, 1):
        width, height = frame["processed_size_wh"]
        dataset["images"].append({"id": image_id, "width": width, "height": height})
        for gt in truth[frame["source_image"]]:
            x1, y1, x2, y2 = gt["bbox_xyxy"]
            annotation = {"id": annotation_id, "image_id": image_id, "category_id": 1,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0,
                          "area": float(gt["mask"].sum()) if gt["mask"] is not None else (x2 - x1) * (y2 - y1)}
            if segmentation:
                annotation["segmentation"] = mask_utils.encode(np.asfortranarray(gt["mask"].astype(np.uint8)))
            dataset["annotations"].append(annotation)
            annotation_id += 1
        for hold in frame["holds"]:
            x1, y1, x2, y2 = hold["bbox_xyxy"]
            prediction = {"image_id": image_id, "category_id": 1,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": hold["detector_score"]}
            if segmentation:
                mask = _read_mask(frame, hold)
                if mask is None:
                    return None
                prediction["segmentation"] = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            predictions.append(prediction)
    if not dataset["annotations"]:
        return None
    with redirect_stdout(StringIO()):
        gt_api = COCO()
        gt_api.dataset = dataset
        gt_api.createIndex()
        if predictions:
            prediction_api = gt_api.loadRes(predictions)
        else:
            prediction_api = COCO()
            prediction_api.dataset = {**dataset, "annotations": []}
            prediction_api.createIndex()
        evaluator = COCOeval(gt_api, prediction_api, "segm" if segmentation else "bbox")
        evaluator.params.maxDets = [1, 10, max_detections]
        evaluator.evaluate()
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, 0, 0, -1]

    def mean_valid(array):
        valid = array[array >= 0]
        return float(valid.mean()) if valid.size else None

    return {"ap50_95": mean_valid(precision), "ap50": mean_valid(precision[0]),
            "max_detections": max_detections,
            "pr_curve_iou50": {"recall": evaluator.params.recThrs.tolist(),
                               "precision": [None if v < 0 else float(v) for v in precision[0]]}}


def evaluate_run(run, gt, threshold, match_iou=0.5, max_detections=500):
    truth = gt["frames"]
    tp = fp = fn = 0
    confusion = np.zeros((6, 7), dtype=int)
    typed_gt_count = sum(g["type"] is not None for values in truth.values() for g in values)
    classification_pairs = []
    mask_ious, matches = [], {}
    for frame in run["frames"]:
        selected = sorted((h for h in frame["holds"] if h["detector_score"] >= threshold),
                          key=lambda h: h["detector_score"], reverse=True)[:max_detections]
        actual = truth[frame["source_image"]]
        matching = detection_matches(selected, actual, match_iou)
        tp += len(matching["pairs"])
        fp += len(matching["fp"])
        fn += len(matching["fn"])
        matches[frame["frame_id"]] = {
            "tp": [selected[p["prediction"]]["hold_id"] for p in matching["pairs"]],
            "fp": [selected[p]["hold_id"] for p in matching["fp"]],
            "fn_boxes": [actual[g]["bbox_xyxy"] for g in matching["fn"]],
        }
        for pair in matching["pairs"]:
            hold, target = selected[pair["prediction"]], actual[pair["truth"]]
            predicted_type = hold.get("predicted_type")
            if target["type"] is not None:
                row = TYPES.index(target["type"])
                column = TYPES.index(predicted_type) if predicted_type in TYPES else 6
                confusion[row, column] += 1
                classification_pairs.append((row, column))
            if target["mask"] is not None:
                mask = _read_mask(frame, hold)
                if mask is not None:
                    union = (mask | target["mask"]).sum()
                    mask_ious.append(float((mask & target["mask"]).sum() / union) if union else 0.0)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    type_report = None
    classifier_available = run["summary"].get("models", {}).get("classifier") != "none"
    if typed_gt_count and classifier_available:
        correct = int(np.trace(confusion[:, :6]))
        per_class = []
        for index, name in enumerate(TYPES):
            class_tp = int(confusion[index, index])
            support, predicted = int(confusion[index].sum()), int(confusion[:, index].sum())
            class_f1 = 2 * class_tp / (support + predicted) if support + predicted else 0.0
            per_class.append({"type": name, "support_on_matched": support, "f1_on_matched": class_f1})
        type_report = {"accuracy_on_matched": correct / len(classification_pairs) if classification_pairs else None,
                       "macro_f1_on_matched_all_six": (sum(c["f1_on_matched"] for c in per_class) / 6
                                                       if classification_pairs else None),
                       "matched_typed_gt": len(classification_pairs), "all_typed_gt": typed_gt_count,
                       "typed_gt_detection_coverage": len(classification_pairs) / typed_gt_count,
                       "end_to_end_correct_type_recall": correct / typed_gt_count,
                       "confusion_matrix": confusion.tolist(), "true_labels": list(TYPES),
                       "predicted_labels": [*TYPES, "unknown"], "per_class": per_class}
    all_gt_masks = all(g["mask"] is not None for values in truth.values() for g in values)
    has_predictions_masks = run["summary"]["segmentation_enabled"]
    gt_count = sum(len(v) for v in truth.values())
    mask_status = ("no_ground_truth_instances" if not gt_count else "available" if has_predictions_masks and all_gt_masks
                   else "model_has_no_masks" if not has_predictions_masks else "ground_truth_masks_incomplete")
    box_ap = _coco_ap(run["frames"], truth, max_detections=max_detections)
    mask_ap = (_coco_ap(run["frames"], truth, segmentation=True, max_detections=max_detections)
               if mask_status == "available" else None)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "box_ap": box_ap, "mask_ap": mask_ap, "mask_status": mask_status,
            "mean_mask_iou_on_box_matched": float(np.mean(mask_ious)) if mask_ious else None,
            "mask_iou_pair_count": len(mask_ious), "classification": type_report, "matches": matches}


def compare_runs(paths: dict[str, Path], thresholds: dict[str, float], annotations: Path | None = None,
                 gt_coordinates="source", match_iou=0.5, max_detections=500) -> dict:
    if not paths:
        raise ValueError("At least one completed run is required")
    if set(paths) != set(thresholds) or any(not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError("Provide one display/evaluation threshold in [0,1] per run")
    if max_detections < 10 or not 0 < match_iou <= 1:
        raise ValueError("max_detections must be >=10 and match_iou must be in (0,1]")
    runs = {name: load_run(path) for name, path in paths.items()}
    validate_shared_frames(runs)
    for name, run in runs.items():
        if thresholds[name] < run["summary"]["settings"]["confidence"]:
            raise ValueError(f"{name}: evaluation threshold is below the saved candidate floor; rerun inference")
    reference = next(iter(runs.values()))["frames"]
    gt = load_ground_truth(annotations, reference, gt_coordinates) if annotations else None
    models = []
    for name, run in runs.items():
        filtered = [h for frame in run["frames"] for h in frame["holds"]
                    if h["detector_score"] >= thresholds[name]]
        model_times = [f["timings"]["model_seconds"] * 1000 for f in run["frames"] if "timings" in f]
        pipeline_times = [f["elapsed_seconds"] * 1000 for f in run["frames"]]
        row = {"model": name, "run_path": run["path"], "images": len(run["frames"]),
               "candidate_count": sum(len(f["holds"]) for f in run["frames"]),
               "detections_at_threshold": len(filtered), "threshold": thresholds[name],
               "needs_review_at_threshold": sum(h["needs_review"] for h in filtered),
               "masks_available": run["summary"]["segmentation_enabled"],
               "model_ms_p50": float(np.median(model_times)) if model_times else None,
               "model_ms_p95": float(np.percentile(model_times, 95)) if model_times else None,
               "pipeline_ms_mean": float(np.mean(pipeline_times)),
               "quality": evaluate_run(run, gt, thresholds[name], match_iou, max_detections) if gt else None}
        models.append(row)
    return {"schema_version": "1.0", "quality_status": "evaluated" if gt else "ground_truth_required",
            "protocol": {"matching": "class-agnostic, confidence-ordered, one-to-one",
                         "match_iou": match_iou, "max_detections": max_detections,
                         "ap_candidate_floor": {n: r["summary"]["settings"]["confidence"] for n, r in runs.items()},
                         "ap_uses_all_saved_candidates": True,
                         "type_metrics": "raw top-1 type on IoU-matched typed GT; unknown counts as incorrect",
                         "timing": "model load excluded; CUDA synchronized for new runs; warmup included"},
            "ground_truth": ({k: v for k, v in gt.items() if k != "frames"} if gt else None),
            "models": models}
