"""Classify hold observations by matching RGB detections to masks."""

import numpy as np
from PIL import Image

from .geometry import box_iou

TYPE_PROPOSAL_FLOOR = 0.2


def assign_type_proposals(holds, proposals, minimum_score=0.5, minimum_iou=0.3):
    """Match detector boxes one-to-one; retain segmentation instances without a type match."""
    from scipy.optimize import linear_sum_assignment
    result = [dict(hold, hold_type="unknown", hold_type_confidence=None,
                   hold_type_source="not_inferred", type_needs_review=True,
                   type_match_iou=None, type_proposals=[],
                   needs_review=True,
                   review_reasons=sorted(set(hold.get("review_reasons", [])) | {"rgb_type_requires_review"}))
              for hold in holds]
    if not holds or not proposals:
        return result
    overlap = np.array([[box_iou(h["bbox_xyxy"], p["bbox_xyxy"]) for p in proposals] for h in holds])
    for row, hold in enumerate(result):
        hold["type_proposals"] = [{"label": proposal["label"], "score": proposal["score"],
                                   "iou": float(overlap[row, column])}
                                  for column, proposal in enumerate(proposals) if overlap[row, column] >= minimum_iou]
    # Gate both score and overlap before the global assignment.
    eligible = (overlap >= minimum_iou) & np.array([p["score"] >= minimum_score for p in proposals])
    valid = np.where(eligible, overlap, 0)
    for row, column in zip(*linear_sum_assignment(-valid)):
        proposal = proposals[column]
        if not eligible[row, column]:
            continue
        result[row].update(hold_type=proposal["label"], hold_type_confidence=proposal["score"],
                           hold_type_source="yolo26_rgb_joint_score", type_needs_review=True,
                           type_match_iou=float(overlap[row, column]))
    return result


def summarize_type_assignments(observations, minimum_score):
    counts = {"no_overlapping_yolo_candidate": 0, "below_type_min_score": 0, "eligible_but_unmatched": 0}
    for observation in observations:
        if observation["hold_type"] != "unknown":
            continue
        proposals = observation["type_proposals"]
        if not proposals:
            counts["no_overlapping_yolo_candidate"] += 1
        elif max(proposal["score"] for proposal in proposals) < minimum_score:
            counts["below_type_min_score"] += 1
        else:
            counts["eligible_but_unmatched"] += 1
    return {"minimum_score": minimum_score, "unknown_observations_by_reason": counts}


def classify_frames(frames, weights, device, imgsz, minimum_score):
    """Attach YOLO type proposals to the existing masks in each frame."""
    from .models import HoldDetector

    detector = HoldDetector(weights, device)
    for frame in frames:
        with Image.open(frame["source_image"]) as image:
            detections = detector.detect(
                image.convert("RGB"), confidence=TYPE_PROPOSAL_FLOOR, imgsz=imgsz
            )
        proposals = [{"bbox_xyxy": list(d.bbox), "score": d.score, "label": d.label}
                     for d in detections]
        frame["holds"] = assign_type_proposals(frame["holds"], proposals, minimum_score=minimum_score)
