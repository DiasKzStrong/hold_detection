"""Download explicitly requested, revision-pinned reference checkpoints."""

import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

REFERENCE_REPOSITORY = "Marcuss2106/indoor-climbing-gym-hold-classification"
REFERENCE_REVISION = "06a763cd8480a8c5cf2b22ec47c749b8bf4fe97d"
ASSETS = {
    "detector": {
        "path": "runs/detect/yolov11s-climbing-gym/weights/best.pt",
        "filename": "climbing-yolo11s.pt",
        "sha256": "5da8ce4bf43c459c618ae0333e136b88759035d2aa43e45ef7a96a4647e15f72",
    },
    "classifier": {
        "path": "sample_checkpoint.pth",
        "filename": "climbing-cnn.pth",
        "sha256": "19ae7c675b2846dddeef8a5eadb0b7ede21cd8023233db2fabeadc6c8bb6aa0a",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_asset(name: str, destination: Path) -> Path:
    asset = ASSETS[name]
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / asset["filename"]
    if target.exists():
        if sha256(target) != asset["sha256"]:
            raise ValueError(f"Checksum mismatch: {target}; existing file was not replaced")
        return target
    url = (
        f"https://raw.githubusercontent.com/{REFERENCE_REPOSITORY}/"
        f"{REFERENCE_REVISION}/{asset['path']}"
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
        if sha256(temporary) != asset["sha256"]:
            raise ValueError(f"Downloaded {name} failed SHA-256 verification")
        # A concurrent download may have completed; never replace an unknown file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if sha256(target) != asset["sha256"]:
                raise ValueError(f"Conflicting file appeared at {target}")
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
