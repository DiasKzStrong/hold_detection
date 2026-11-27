"""Shared image discovery and HEIF support."""

from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener(thumbnails=False)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


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


def processed_image(frame):
    """Read the same RGB orientation used by frame-local detections and masks."""
    with Image.open(frame["source_image"]) as original:
        image = original.convert("RGB")
    rotation = frame.get("rotation_clockwise_degrees", 0)
    return image.rotate(-rotation, expand=True) if rotation else image
