"""Portable input/bundle helpers used by the standalone Colab notebook."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

from .gallery import build_gallery
from .pipeline import IMAGE_EXTENSIONS, collect_images
from .polycam import inspect_archive, prepare_archive


@dataclass(frozen=True)
class PreparedInput:
    source: Path
    kind: str
    image_count: int


def photo_directory(path: Path) -> tuple[Path, str]:
    for group in ("corrected_images", "images"):
        candidate = path / "keyframes" / group
        if candidate.is_dir():
            return candidate, "polycam"
    if path.name in {"corrected_images", "images"} and path.parent.name == "keyframes":
        return path, "polycam"
    return path, "photos"


def prepare_inputs(paths: list[Path], workspace: Path) -> PreparedInput:
    """Accept a photo folder, individual photos, a photo ZIP, or a Polycam ZIP.

    Never mix depth/confidence PNGs or texture atlases into the RGB input set.
    Extract/copy operations require a fresh destination and retain the originals.
    """
    if not paths or any(not p.exists() for p in paths):
        raise ValueError("Select existing JPEG/PNG photos, a photo directory, or a raw-data ZIP")
    if len(paths) == 1 and paths[0].is_dir():
        source, kind = photo_directory(paths[0])
        if source == paths[0] and any((source / name).is_dir() for name in ("depth", "confidence")):
            raise ValueError("Choose keyframes/corrected_images, not a mixed depth/RGB directory")
        return PreparedInput(source, kind, len(collect_images(source)))
    archives = [p for p in paths if p.suffix.lower() == ".zip"]
    if archives:
        if len(paths) != 1:
            raise ValueError("Upload one ZIP OR several photos, not both")
        archive = archives[0]
        info = inspect_archive(archive)  # Validates paths, duplicates, symlinks and total size.
        if info["complete_frame_ids"]:
            prepare_archive(archive, workspace)
            source, kind = photo_directory(workspace)
            return PreparedInput(source, kind, len(collect_images(source)))
        with ZipFile(archive) as z:
            files = [n for n in z.namelist() if not n.endswith("/")]
            group = next((g for g in ("corrected_images", "images") if info["groups"][g]), None)
            if group:
                selected = [n for n in files if PurePosixPath(n).parent == PurePosixPath("keyframes") / group
                            and PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS]
                kind = "polycam"
            else:
                if any(info["groups"][g] for g in ("depth", "confidence")):
                    raise ValueError("Archive has depth/confidence maps but no RGB keyframes")
                if any(PurePosixPath(n).suffix.lower() in {".obj", ".glb", ".gltf"} for n in files):
                    raise ValueError("Upload raw_data.zip or photos, not a mesh/texture export")
                selected = [n for n in files if PurePosixPath(n).suffix.lower() in IMAGE_EXTENSIONS
                            and "__MACOSX" not in PurePosixPath(n).parts
                            and not PurePosixPath(n).name.startswith("._")]
                kind = "photos"
            if not selected:
                raise ValueError("No supported RGB photos in ZIP; use JPEG or PNG")
            workspace.mkdir(parents=True, exist_ok=False)
            for name in selected:
                target = workspace / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        source, kind = photo_directory(workspace)
        return PreparedInput(source, kind, len(collect_images(source)))
    if any(not p.is_file() or p.suffix.lower() not in IMAGE_EXTENSIONS for p in paths):
        raise ValueError("Supported uploads: JPEG, PNG, WebP, BMP, TIFF, or one ZIP")
    workspace.mkdir(parents=True, exist_ok=False)
    unique_names = len({p.name for p in paths}) == len(paths)
    for index, path in enumerate(paths):
        # Preserve image filenames so COCO annotations can match uploaded photos.
        name = path.name if unique_names else f"{index:04d}_{path.name}"
        shutil.copy2(path, workspace / name)
    return PreparedInput(workspace, "photos", len(paths))


def bundle_results(run: Path, destination: Path) -> Path:
    if destination.resolve().is_relative_to(run.resolve()):
        raise ValueError("Save the result ZIP outside the run directory")
    build_gallery(run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as z:
        for path in sorted(run.rglob("*")):
            if path.is_symlink():
                raise ValueError("Do not bundle symlinks in a result directory")
            if path.is_file():
                if not path.resolve().is_relative_to(run.resolve()):
                    raise ValueError("Result file escapes its run directory")
                z.write(path, arcname=run.name + "/" + path.relative_to(run).as_posix())
    return destination
