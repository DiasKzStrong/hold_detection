"""Prepare immutable single-photo, multi-photo and video wall captures."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from .pipeline import collect_images

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def source_identity(source):
    source = Path(source).resolve()
    paths = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError(f"Input does not exist or is empty: {source}")
    return hashlib.sha256(json.dumps([
        (str(p.relative_to(source)) if source.is_dir() else p.name, p.stat().st_size, p.stat().st_mtime_ns)
        for p in paths
    ]).encode()).hexdigest()


def prepare_capture(source, output, mode="2d", max_frames=48, video_stride_seconds=0.75, max_image_edge=1600):
    """Copy oriented RGB frames; 3D ordinary inputs never require a prebuilt GLB."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if mode not in {"2d", "3d"} or max_frames < 2 or video_stride_seconds <= 0 or max_image_edge < 64:
        raise ValueError("Choose mode 2d/3d, max_frames >= 2 and a positive video stride.")
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Output must be outside the input directory.")
    signature = {"source": str(source), "identity": source_identity(source), "mode": mode,
                 "max_frames": max_frames, "video_stride_seconds": video_stride_seconds,
                 "max_image_edge": max_image_edge}
    marker = output / "capture.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["settings"] != signature:
            raise ValueError("Capture settings/input changed; choose a new output.")
        if not all((output / item["path"]).is_file() for item in manifest["frames"]):
            raise ValueError("Prepared capture is incomplete; choose a new output.")
        return manifest
    settings_file = output / "capture_settings.json"
    if output.exists():
        if not settings_file.is_file() or json.loads(settings_file.read_text()) != signature:
            raise FileExistsError("Incomplete capture output has different/unknown settings; choose a new output")
    output.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(json.dumps(signature, indent=2) + "\n")
    images = output / "images"
    images.mkdir(exist_ok=True)
    frames = []

    def save(image, origin, timestamp=None):
        path = images / f"frame_{len(frames):05d}.jpg"
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        oriented.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
        oriented.save(path, quality=97)
        frames.append({"path": path.relative_to(output).as_posix(), "source": str(origin),
                       "timestamp_seconds": timestamp})

    if source.suffix.lower() in VIDEO_EXTENSIONS:
        if mode != "3d":
            raise ValueError("2D mode accepts one photograph; use 3D mode for video.")
        import cv2
        video = cv2.VideoCapture(str(source))
        try:
            fps, count = video.get(cv2.CAP_PROP_FPS), video.get(cv2.CAP_PROP_FRAME_COUNT)
            if not video.isOpened() or fps <= 0 or count <= 0:
                raise ValueError("Cannot read the video frame rate/count.")
            import numpy as np
            indices = np.unique(np.linspace(0, max(0, count - 1), min(max_frames, max(2, int(count / fps / video_stride_seconds))), dtype=int))
            for index in indices:
                video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = video.read()
                if ok:
                    save(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), source, round(index / fps, 4))
        finally:
            video.release()
        kind = "video"
    else:
        if source.suffix.lower() == ".zip":
            from .colab_support import prepare_inputs
            prepared = prepare_inputs([source], output / "extracted")
            paths = collect_images(prepared.source)
        else:
            paths = collect_images(source)
        if mode == "2d" and len(paths) != 1:
            raise ValueError("2D mode requires exactly one photo.")
        if len(paths) > max_frames:
            import numpy as np
            paths = [paths[i] for i in np.linspace(0, len(paths) - 1, max_frames, dtype=int)]
        for path in paths:
            with Image.open(path) as image:
                save(image, path)
        kind = "photos"
    if mode == "3d" and len(frames) < 3:
        raise ValueError("3D reconstruction requires at least three overlapping views; more views are recommended.")
    manifest = {"schema_version": "1.0", "settings": signature, "input_kind": kind,
                "image_source": str(images), "frames": frames}
    marker.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
