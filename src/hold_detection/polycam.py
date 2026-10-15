"""Validate and unpack the subset of a Polycam export required for perception."""

import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

GROUPS = ("images", "corrected_images", "cameras", "corrected_cameras", "depth", "confidence")
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3


def inspect_archive(archive: Path) -> dict:
    with ZipFile(archive) as z:
        members = z.infolist()
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Polycam archive exceeds the 4 GiB uncompressed limit")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(f"Archive symlinks are not supported: {member.filename}")
        files = [m.filename for m in members if not m.is_dir()]
        if len(set(files)) != len(files):
            raise ValueError("Archive contains duplicate file paths")
        frames = {
            group: {PurePosixPath(n).stem for n in files
                    if PurePosixPath(n).parent == PurePosixPath("keyframes") / group}
            for group in GROUPS
        }
        complete = sorted(set.intersection(*frames.values()))
        return {
            "archive": str(archive.resolve()), "file_count": len(files),
            "groups": {g: len(f) for g, f in frames.items()},
            "complete_frame_ids": complete,
            "other_files": [n for n in files if not n.startswith("keyframes/")],
        }


def prepare_archive(archive: Path, output: Path) -> dict:
    info = inspect_archive(archive)
    if not info["complete_frame_ids"]:
        raise ValueError("No frames with RGB, camera, depth and confidence data")
    output.mkdir(parents=True, exist_ok=False)
    with ZipFile(archive) as z:
        for name in z.namelist():
            path = PurePosixPath(name)
            if ((len(path.parts) == 3 and path.parts[0] == "keyframes"
                 and path.parts[1] in GROUPS)
                or name in {"mesh_info.json", "thumbnail.jpg"}):
                member = z.getinfo(name)
                if member.is_dir():
                    continue
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
    (output / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
