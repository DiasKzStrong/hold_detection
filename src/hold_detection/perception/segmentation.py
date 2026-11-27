"""Resumable per-frame inference for a prepared wall capture."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..capture.images import collect_images
from ..review.gallery import build_gallery
from .assets import sha256
from .color import measure_frame_colors
from .hold_types import TYPE_PROPOSAL_FLOOR, classify_frames


class PipelinePaused(Exception):
    """The current stage can be resumed from completed frame/stage artifacts."""


def segment_capture(capture, run, reconstruction_path=None, sam_weights=None, type_weights=None,
                    device="0", confidence=0.5, stop_file=None, type_imgsz=1024,
                    type_min_score=0.5):
    """Run class-agnostic SAM3, one frame per process, then attach YOLO26 RGB type proposals."""
    capture, run = Path(capture).resolve(), Path(run).resolve()
    manifest = json.loads((capture / "capture.json").read_text())
    reconstruction = json.loads(Path(reconstruction_path).read_text()) if reconstruction_path else None
    if manifest["settings"]["mode"] == "3d" and not reconstruction:
        raise ValueError("Run reconstruction before segmenting a 3D capture")
    source = Path(reconstruction["image_source"] if reconstruction else manifest["image_source"])
    weights = Path(sam_weights or "models/sam3/sam3.pt").resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing SAM3 checkpoint: {weights}; run holds fetch-sam3 first")
    if type_weights and not Path(type_weights).is_file():
        raise FileNotFoundError(type_weights)
    if type_weights and type_imgsz <= 0:
        raise ValueError("Hold-type inference image size must be positive")
    if type_weights and not TYPE_PROPOSAL_FLOOR <= type_min_score <= 1:
        raise ValueError(f"Hold-type minimum score must be between {TYPE_PROPOSAL_FLOOR} and 1")
    settings = {"capture": manifest["settings"], "source": str(source), "confidence": confidence,
                "sam_sha256": sha256(weights),
                "type_sha256": sha256(Path(type_weights)) if type_weights else None,
                "type_imgsz": type_imgsz if type_weights else None}
    run.mkdir(parents=True, exist_ok=True)
    specification = run / "analysis_settings.json"
    if specification.exists():
        previous = json.loads(specification.read_text())
        # Runs created before this option used 1024 for every hold-type checkpoint.
        previous.setdefault("type_imgsz", 1024 if previous["type_sha256"] else None)
        if previous != settings:
            raise ValueError("Analysis input/models/settings changed; choose another run directory")
    specification.write_text(json.dumps(settings, indent=2) + "\n")
    inference = run / "inference"
    inference.mkdir(exist_ok=True)
    batches, logs = run / "frame_batches", run / "logs"
    batches.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    stop_file = Path(stop_file) if stop_file else run / "STOP_REQUESTED"
    frame_summaries = []
    for index, image_path in enumerate(collect_images(source)):
        if stop_file.exists():
            raise PipelinePaused("Paused between frames; completed frames will be reused")
        batch = batches / f"frame_{index:05d}"
        summary_path = batch / "summary.json"
        reused = summary_path.is_file()
        if not reused:
            if batch.exists():
                # Preserve an interrupted frame for inspection; retry it in a fresh directory.
                batch.rename(batches / f"incomplete_{index:05d}_{time.time_ns()}")
            command = [sys.executable, "-m", "hold_detection.cli", "sam3", str(source),
                       "--output", str(batch), "--weights", str(weights), "--device", str(device),
                       "--start", str(index), "--limit", "1", "--no-finalize",
                       "--prompt", "climbing hold", "--volume-prompt", "climbing volume",
                       "--conf", str(confidence), "--classifier", "none"]
            with (logs / f"sam3_{index:05d}.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"SAM3 failed for {image_path.name}; inspect {logs / f'sam3_{index:05d}.log'}")
        summary = json.loads(summary_path.read_text())
        frame_summaries.extend(summary["frames"])
        for frame_dir in sorted(p.parent for p in batch.glob("*/holds.json")):
            destination = inference / frame_dir.name
            if not destination.exists():
                shutil.copytree(frame_dir, destination)
        print(f"SAM3 {index + 1}: {image_path.name} {'reused' if reused else 'saved'}", flush=True)
    summaries = {"schema_version": "1.0", "frame_count": len(frame_summaries),
                 "detection_count": sum(frame["hold_count"] for frame in frame_summaries),
                 "segmentation_enabled": True,
                 "frames": frame_summaries, "grip_type_inference_enabled": bool(type_weights),
                 "settings": {"rotate": 0}, "models": settings}
    if manifest["settings"]["mode"] == "2d" and manifest["frames"][0].get("input_crop"):
        summaries["input_crop"] = manifest["frames"][0]["input_crop"]
    summary_path = inference / "summary.json"
    completion = run / "SEGMENTATION_COMPLETE"
    completion.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)
    frame_paths = sorted(inference.glob("*/holds.json"))
    frames = [json.loads(path.read_text()) for path in frame_paths]
    measure_frame_colors(frames)
    if type_weights:
        classify_frames(frames, Path(type_weights), device, type_imgsz, type_min_score)
        summaries["settings"]["type_min_score"] = type_min_score
    for path, frame in zip(frame_paths, frames, strict=True):
        for hold in frame["holds"]:
            hold.pop("route_candidate_id", None)
        path.write_text(json.dumps(frame, indent=2, allow_nan=False) + "\n")
        (path.parent / "routes.jpg").unlink(missing_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    build_gallery(inference, include_scene=False)
    completion.touch()
    return str(inference)
