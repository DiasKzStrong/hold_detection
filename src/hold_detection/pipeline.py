"""Batch inference and inspectable, versioned per-frame artifacts."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .geometry import TYPES, crop_box

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
COLORS = dict(zip(TYPES, ["#f5b041", "#58d68d", "#5dade2", "#ec7063", "#af7ac5", "#f4d03f"]))


def collect_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            return images
    raise ValueError(f"No supported images at {source}; point to corrected_images, not a raw export root")


def validate_image_source(source: Path):
    if source.is_dir() and ((source / "keyframes").is_dir() or (source / "mesh_info.json").is_file()):
        raise ValueError("Use <export>/keyframes/corrected_images, not the export root")


def make_hold_record(detection, index, scores, minimum_score, mask_info=None,
                     mask_method="sam2_box_prompt", infer_grip_types=True):
    if not infer_grip_types:
        label, score, method, alternatives = None, None, "disabled", None
    elif scores is None and not detection.label_is_type:
        label, score, method, alternatives = "unknown", None, "not_inferred", None
    elif scores is None:
        label, score = detection.label, detection.score
        method = "detector_joint_class_score"
        alternatives = None  # YOLO postprocessing does not expose the full type distribution.
    else:
        label = max(scores, key=scores.get)
        score = scores[label]
        method = "crop_cnn_softmax"
        alternatives = scores
    unknown = infer_grip_types and (score is None or score < minimum_score)
    disagreement = (infer_grip_types and scores is not None
                    and detection.label_is_type and label != detection.label)
    mask, mask_score = mask_info if mask_info is not None else (None, None)
    empty_mask = mask is not None and not mask.any()
    return {
        "hold_id": f"h{index:04d}", "bbox_xyxy": list(detection.bbox),
        "bbox_center_xy": [(detection.bbox[0] + detection.bbox[2]) / 2,
                           (detection.bbox[1] + detection.bbox[3]) / 2],
        "detector_label": detection.label, "detector_score": detection.score,
        "predicted_type": label,
        "object_kind": ("volume" if detection.label == "volume" else "hold"
                        if not infer_grip_types else
                        detection.label if not detection.label_is_type
                        and scores is None and detection.label in {"hold", "volume"}
                        else "unknown" if unknown else "volume" if label == "volume" else "hold"),
        "grip_type": (None if not infer_grip_types else
                      "unknown" if unknown else (None if label == "volume" else label)),
        "type_score": score, "type_score_method": method, "type_scores": alternatives,
        "scores_calibrated": False,
        "needs_review": unknown or empty_mask or disagreement,
        "review_reasons": (([] if not infer_grip_types else
                            ["type_not_inferred"] if score is None else
                            ["low_type_score"] if unknown else []))
        + (["empty_mask"] if empty_mask else [])
        + (["classifier_detector_disagreement"] if disagreement else []),
        "mask_path": None, "segmentation_score": mask_score,
        "segmentation_method": mask_method if mask is not None else None,
        "route_id": None, "position_3d": None,
    }


def run_inference(source: Path, output: Path, detector, classifier=None, segmenter=None,
                  confidence=0.35, minimum_type_score=0.5, iou=0.5, imgsz=1024,
                  tile_size=0, overlap=0.25, max_det=500, rotate=0, limit=None,
                  model_metadata=None, infer_grip_types=False) -> dict:
    validate_image_source(source)
    if source.is_dir() and output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Output must be outside the input image directory")
    images = collect_images(source)
    if limit is not None:
        images = images[:limit]
    # Never silently overwrite a previous run or user files.
    output.mkdir(parents=True, exist_ok=False)
    frames = []
    started = time.monotonic()
    for path in images:
        image_started = time.monotonic()
        with Image.open(path) as original:
            original_size = original.size
            image = original.convert("RGB")
        if rotate:
            image = image.rotate(-rotate, expand=True)
        def synchronize():
            if str(getattr(detector, "device", "cpu")).startswith("cuda"):
                import torch
                torch.cuda.synchronize(detector.device)

        synchronize()
        stage_started = time.monotonic()
        detections = detector.detect(
            image, confidence=confidence, iou=iou, imgsz=imgsz,
            tile_size=tile_size, overlap=overlap, max_det=max_det,
        )
        synchronize()
        detection_seconds = time.monotonic() - stage_started
        stage_started = time.monotonic()
        predictions = classifier.classify(image, detections) if classifier else [None] * len(detections)
        synchronize()
        classification_seconds = time.monotonic() - stage_started
        stage_started = time.monotonic()
        masks = segmenter.segment(image, detections) if segmenter else [None] * len(detections)
        synchronize()
        segmentation_seconds = time.monotonic() - stage_started
        if len(predictions) != len(detections) or len(masks) != len(detections):
            raise ValueError("Model returned a different number of results than detections")
        identity = path.name if source.is_file() else path.relative_to(source).as_posix()
        frame_id = f"{path.stem}-{hashlib.sha256(identity.encode()).hexdigest()[:8]}"
        frame_dir = output / frame_id
        frame_dir.mkdir()
        crops_dir = frame_dir / "crops"
        crops_dir.mkdir()
        annotation = image.copy()
        records = []
        for index, (detection, scores, mask_info) in enumerate(zip(detections, predictions, masks), 1):
            record = make_hold_record(detection, index, scores, minimum_type_score, mask_info,
                                      getattr(segmenter, "method", "sam2_box_prompt"),
                                      infer_grip_types=infer_grip_types)
            if scores is not None and getattr(classifier, "score_method", None):
                record["type_score_method"] = classifier.score_method
            if mask_info is not None:
                record["segmentation_score_method"] = getattr(segmenter, "score_method", "predicted_mask_iou")
            if mask_info is not None:
                mask, _ = mask_info
                if mask.shape != (image.height, image.width):
                    raise ValueError("SAM mask dimensions differ from the processed image")
                (frame_dir / "masks").mkdir(exist_ok=True)
                mask_path = Path("masks") / f"{record['hold_id']}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(frame_dir / mask_path)
                record["mask_path"] = mask_path.as_posix()
                overlay_color = (
                    COLORS.get(record["predicted_type"], "#b0b0b0")
                    if infer_grip_types else "#3b82f6"
                )
                overlay = Image.new("RGB", image.size, overlay_color)
                alpha = Image.fromarray(mask.astype(np.uint8) * 65)
                annotation = Image.composite(overlay, annotation, alpha)
                pixels = np.asarray(image)[mask]
                record["median_rgb"] = np.median(pixels, axis=0).round().astype(int).tolist() if len(pixels) else None
                record["color_method"] = "mask_median_rgb"  # Not route membership.
            crop_path = Path("crops") / f"{record['hold_id']}.jpg"
            crop_padding = getattr(classifier, "metadata", {}).get("crop_padding", 0.1)
            image.crop(crop_box(detection.bbox, *image.size, padding=crop_padding)).save(frame_dir / crop_path, quality=95)
            record["crop_path"] = crop_path.as_posix()
            records.append(record)
        draw = ImageDraw.Draw(annotation)
        for record in records:
            color = (COLORS.get(record["predicted_type"], "#b0b0b0")
                     if infer_grip_types else "#3b82f6")
            draw.rectangle(record["bbox_xyxy"], outline=color, width=3)
            x, y = record["bbox_xyxy"][:2]
            label = record["grip_type"] or record["object_kind"]
            score_text = (f"type {record['type_score']:.2f}" if record["type_score"] is not None
                          else f"det {record['detector_score']:.2f}")
            text = f"{record['hold_id']} {label} {score_text}"
            bounds = draw.textbbox((x, max(0, y - 13)), text)
            draw.rectangle(bounds, fill="black")
            draw.text((x, max(0, y - 13)), text, fill=color)
        annotation.save(frame_dir / "annotated.jpg", quality=95)
        frame = {
            "schema_version": "1.0", "frame_id": frame_id,
            "source_image": str(path.resolve()), "source_size_wh": list(original_size),
            "processed_size_wh": list(image.size), "rotation_clockwise_degrees": rotate,
            "coordinate_space": "processed_image_pixels",
            "camera_registration_verified": False,
            "hold_ids_are": "frame_local_not_multiview_tracks",
            "timings": {"detection_seconds": detection_seconds,
                        "classification_seconds": classification_seconds,
                        "segmentation_seconds": segmentation_seconds,
                        "model_seconds": detection_seconds + classification_seconds + segmentation_seconds},
            "holds": records, "elapsed_seconds": round(time.monotonic() - image_started, 3),
        }
        (frame_dir / "holds.json").write_text(json.dumps(frame, indent=2, allow_nan=False) + "\n")
        counts = dict(Counter(r["grip_type"] or r["object_kind"] for r in records))
        frames.append({"frame_id": frame_id, "source_image": str(path),
                       "hold_count": len(records), "objects": counts,
                       "needs_review": sum(r["needs_review"] for r in records)})
        print(f"{path.name}: {len(records)} detections; {counts}", flush=True)
    summary = {
        "schema_version": "1.0", "models": model_metadata or {},
        "settings": {"confidence": confidence, "minimum_type_score": minimum_type_score,
                     "nms_iou": iou, "imgsz": imgsz, "tile_size": tile_size,
                     "tile_overlap": overlap, "max_det": max_det, "rotate": rotate,
                     "crop_padding": getattr(classifier, "metadata", {}).get("crop_padding", 0.1)},
        "frame_count": len(frames), "detection_count": sum(f["hold_count"] for f in frames),
        "segmentation_enabled": segmenter is not None,
        "grip_type_inference_enabled": infer_grip_types,
        "timing_protocol": "CUDA synchronized; model loading excluded; first-image warmup included; frame elapsed includes image/crop export",
        "elapsed_seconds": round(time.monotonic() - started, 3), "frames": frames,
        "evaluation": "Unlabeled inference; hold recall, mask quality and route accuracy have not been measured.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    from .scene import build_scene
    build_scene(output)
    from .gallery import build_gallery
    build_gallery(output)
    return summary
