"""Resumable stages for wall analysis; model inference runs in fresh processes per frame."""

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .assets import sha256
from .pipeline import collect_images


class PipelinePaused(Exception):
    """The current stage can be resumed from completed frame/stage artifacts."""


def segment_capture(capture, run, reconstruction_path=None, sam_weights=None, type_weights=None,
                    device="0", confidence=0.5, stop_file=None):
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
    settings = {"capture": manifest["settings"], "source": str(source), "confidence": confidence,
                "sam_sha256": sha256(weights),
                "type_sha256": sha256(Path(type_weights)) if type_weights else None}
    run.mkdir(parents=True, exist_ok=True)
    specification = run / "analysis_settings.json"
    if specification.exists() and json.loads(specification.read_text()) != settings:
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
        if not summary_path.is_file():
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
        print(f"SAM3 {index + 1}: {image_path.name} saved", flush=True)
    summaries = {"schema_version": "1.0", "frame_count": len(frame_summaries),
                 "detection_count": sum(frame["hold_count"] for frame in frame_summaries),
                 "segmentation_enabled": True,
                 "frames": frame_summaries, "grip_type_inference_enabled": bool(type_weights),
                 "settings": {"rotate": 0}, "models": settings}
    (inference / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    from .scene import build_scene
    build_scene(inference)
    if type_weights:
        assign_run_types(inference, Path(type_weights), device)
    from .gallery import build_gallery
    build_gallery(inference)
    (run / "SEGMENTATION_COMPLETE").touch()
    return str(inference)


def assign_run_types(inference, weights, device="0"):
    """Keep SAM identities and masks; assign six-class detections through overlap."""
    from .models import HoldDetector
    from .wall_scene import assign_type_proposals
    from PIL import Image
    detector = HoldDetector(Path(weights), device)
    for path in sorted(Path(inference).glob("*/holds.json")):
        frame = json.loads(path.read_text())
        with Image.open(frame["source_image"]) as image:
            detections = detector.detect(image.convert("RGB"), confidence=0.2, imgsz=1024)
        proposals = [{"bbox_xyxy": list(d.bbox), "score": d.score, "label": d.label} for d in detections]
        frame["holds"] = assign_type_proposals(frame["holds"], proposals)
        path.write_text(json.dumps(frame, indent=2, allow_nan=False) + "\n")


def analyze_wall(source, output, mode="2d", sam_weights=None, type_weights=None,
                 device="0", max_frames=48, colmap="colmap", scale_m_per_unit=None,
                 quality="medium", annotations=None, confidence=0.5, scale_reference=None):
    """Run the complete photo/video workflow; resume completed preparation and reconstruction."""
    from .capture import prepare_capture
    from .reconstruction import reconstruct_wall
    from .wall_scene import export_wall_scene
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop = output / "STOP_REQUESTED"
    if stop.exists():
        stop.unlink()

    def request_pause(signum, frame):
        stop.touch()

    previous = {sig: signal.signal(sig, request_pause) for sig in {signal.SIGTERM, signal.SIGUSR1}}
    try:
        capture = output / "capture"
        prepare_capture(source, capture, mode=mode, max_frames=max_frames)
        if stop.exists():
            raise PipelinePaused("Paused after input preparation")
        reconstruction_path = None
        if mode == "3d":
            reconstruct_wall(capture, output / "reconstruction", colmap, quality, scale_m_per_unit, scale_reference)
            reconstruction_path = output / "reconstruction/reconstruction.json"
        segment_capture(capture, output, reconstruction_path, sam_weights, type_weights, device, confidence, stop)
        return export_wall_scene(output, mode, reconstruction_path, annotations)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
