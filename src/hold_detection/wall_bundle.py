"""Discover and prepare one wall capture from the project dataset layout."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from zipfile import is_zipfile

from .polycam import GROUPS, inspect_archive, prepare_archive


ROUTE_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def _matches(directory: Path, predicate) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if predicate(path)),
        key=lambda path: path.name.lower(),
    )


def _inspect_capture(directory: Path) -> dict:
    keyframes = directory / "keyframes"
    frames = {
        group: {path.stem for path in (keyframes / group).glob("*") if path.is_file()}
        for group in GROUPS
    }
    complete = sorted(set.intersection(*frames.values()))
    return {
        "archive": None,
        "directory": str(directory.resolve()),
        "file_count": sum(len(items) for items in frames.values()),
        "groups": {group: len(items) for group, items in frames.items()},
        "complete_frame_ids": complete,
        "other_files": [],
    }


def inspect_wall_bundle(directory: Path) -> dict:
    """Describe a wall folder without modifying or unpacking it."""
    if not directory.is_dir():
        raise ValueError(f"Wall bundle is not a directory: {directory}")
    raw_sources = _matches(
        directory,
        lambda path: "raw_data" in path.name.lower()
        and (
            (path.is_file() and is_zipfile(path))
            or (path.is_dir() and (path / "keyframes").is_dir())
        ),
    )
    glb_files = _matches(
        directory,
        lambda path: path.is_file() and path.suffix.lower() == ".glb",
    )
    obj_backups = _matches(
        directory,
        lambda path: (
            path.is_dir() and "obj" in path.name.lower()
        ) or (
            path.is_file()
            and path.suffix.lower() == ".zip"
            and "obj" in path.stem.lower()
        ),
    )
    photo_directories = _matches(
        directory,
        lambda path: path.is_dir() and path.name.lower() == "photos",
    )
    route_photos = _matches(
        directory,
        lambda path: path.is_file() and path.suffix.lower() in ROUTE_PHOTO_EXTENSIONS,
    )
    for photo_directory in photo_directories:
        route_photos.extend(_matches(
            photo_directory,
            lambda path: path.is_file() and path.suffix.lower() in ROUTE_PHOTO_EXTENSIONS,
        ))
    route_notes = _matches(
        directory,
        lambda path: path.is_file()
        and path.stem.lower() == "route_notes"
        and path.suffix.lower() in {".txt", ".md", ".json"},
    )
    problems = []
    if len(raw_sources) != 1:
        problems.append(f"expected exactly one raw_data ZIP or directory, found {len(raw_sources)}")
    if len(glb_files) != 1:
        problems.append(f"expected exactly one GLB, found {len(glb_files)}")
    raw_info = None
    if len(raw_sources) == 1:
        raw_info = (
            inspect_archive(raw_sources[0])
            if raw_sources[0].is_file()
            else _inspect_capture(raw_sources[0])
        )
    route_notes_text = route_notes[0].read_text() if len(route_notes) == 1 else None
    return {
        "wall_id": directory.name,
        "directory": str(directory.resolve()),
        "raw_source": str(raw_sources[0].resolve()) if len(raw_sources) == 1 else None,
        "raw_source_type": (
            "zip" if len(raw_sources) == 1 and raw_sources[0].is_file()
            else "directory" if len(raw_sources) == 1 else None
        ),
        "raw_archive": (
            str(raw_sources[0].resolve())
            if len(raw_sources) == 1 and raw_sources[0].is_file() else None
        ),
        "raw_archive_info": raw_info,
        "glb": str(glb_files[0].resolve()) if len(glb_files) == 1 else None,
        "obj_backups": [str(path.resolve()) for path in obj_backups],
        "route_photos": [str(path.resolve()) for path in route_photos],
        "route_notes": str(route_notes[0].resolve()) if len(route_notes) == 1 else None,
        "route_notes_text": route_notes_text,
        "problems": problems,
        "ready": not problems and bool(raw_info and raw_info["complete_frame_ids"]),
    }


def prepare_wall_bundle(directory: Path, output: Path) -> dict:
    """Unpack RGB-D capture data and copy the GLB into a portable workspace."""
    info = inspect_wall_bundle(directory)
    if not info["ready"]:
        details = "; ".join(info["problems"]) or "raw archive has no complete RGB-D frames"
        raise ValueError(f"Wall bundle is incomplete: {details}")
    output.mkdir(parents=True, exist_ok=False)
    capture = output / "capture"
    raw_source = Path(info["raw_source"])
    if info["raw_source_type"] == "zip":
        prepare_archive(raw_source, capture)
    else:
        if any(path.is_symlink() for path in raw_source.rglob("*")):
            raise ValueError("Raw-data directories containing symlinks are not supported")
        capture.mkdir()
        for group in GROUPS:
            shutil.copytree(raw_source / "keyframes" / group, capture / "keyframes" / group)
        for name in ("mesh_info.json", "thumbnail.jpg"):
            if (raw_source / name).is_file():
                shutil.copy2(raw_source / name, capture / name)
        (capture / "manifest.json").write_text(
            json.dumps(info["raw_archive_info"], indent=2) + "\n"
        )
    glb_target = output / Path(info["glb"]).name
    shutil.copy2(info["glb"], glb_target)
    manifest = {
        **info,
        "prepared_capture": str(capture.resolve()),
        "prepared_rgb_source": str((capture / "keyframes/corrected_images").resolve()),
        "prepared_glb": str(glb_target.resolve()),
    }
    if not Path(manifest["prepared_rgb_source"]).is_dir():
        manifest["prepared_rgb_source"] = str((capture / "keyframes/images").resolve())
    (output / "wall_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest
