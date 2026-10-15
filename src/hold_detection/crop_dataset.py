"""Prepare auditable, source-split-preserving crops from YOLO or COCO exports."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

from PIL import Image

from .geometry import TYPES, canonical_type, crop_box
from .pipeline import IMAGE_EXTENSIONS
from .polycam import inspect_archive


def read_json(path):
    return json.loads(Path(path).read_text())


def manifest_digest(root):
    return hashlib.sha256((Path(root) / "manifest.json").read_bytes()).hexdigest()


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe dataset path: {name}")
    return path


def load_crops(root):
    root = Path(root).resolve()
    data = read_json(root / "manifest.json")
    if data.get("schema") != "hold-crops-v1" or data.get("classes") != list(TYPES):
        raise ValueError("Expected a prepared six-class hold-crops-v1 manifest")
    seen = set()
    for row in data["samples"]:
        safe_path(root, row["path"])
        if row["path"] in seen or row["split"] not in {"train", "val", "test"}:
            raise ValueError("Duplicate crop or invalid split in manifest")
        if not isinstance(row["label"], int) or not 0 <= row["label"] < len(TYPES):
            raise ValueError("Invalid class ID in crop manifest")
        seen.add(row["path"])
    return data


def _layout(root):
    candidates = [
        root,
        *sorted(root.rglob("Final_Dataset")),
        *sorted(root.rglob("Final_dataset")),
    ]
    candidates += [p.parent for p in root.rglob("train") if p.is_dir()]
    matches = []
    for p in candidates:
        if (
            (p / "train/images").is_dir()
            or (p / "images/train").is_dir()
            or (p / "train/_annotations.coco.json").is_file()
        ) and p not in matches:
            matches.append(p)
    preferred = [p for p in matches if p.name.lower() == "final_dataset"]
    matches = preferred or matches
    if len(matches) != 1:
        raise ValueError(
            "Select one labeled export root with train/images and valid/images (or images/train)."
        )
    return matches[0]


def _yolo_names(root, class_names):
    if class_names:
        names = list(class_names)
    else:
        import yaml

        configs = [
            root / name
            for name in ("data.yaml", "dataset.yaml", "data.yml")
            if (root / name).is_file()
        ]
        if len(configs) != 1:
            raise ValueError(
                "YOLO exports require data.yaml, or explicit --class-names in annotation ID order."
            )
        names = yaml.safe_load(configs[0].read_text()).get("names")
        if isinstance(names, dict):
            names = {int(k): v for k, v in names.items()}
            if set(names) != set(range(len(names))):
                raise ValueError("YOLO class IDs must be contiguous from zero")
            names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or len(names) != 6:
        raise ValueError("Expected six YOLO class names")
    names = [canonical_type(n) for n in names]
    if set(names) != set(TYPES):
        raise ValueError(
            "Expected the six supported hold classes, without duplicate names"
        )
    return names


def _box(box, width, height):
    if len(box) != 4 or not all(math.isfinite(x) for x in box):
        raise ValueError("Invalid annotation box")
    x1, y1, x2, y2 = box
    # Normalized YOLO coordinates can round a boundary by a fraction of a pixel.
    if (
        x1 < -0.5
        or y1 < -0.5
        or x2 > width + 0.5
        or y2 > height + 0.5
        or x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(f"Annotation box is outside the image: {box}")
    return [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)]


def _records(root, class_names):
    names = None
    for split, aliases in [
        ("train", ["train"]),
        ("val", ["valid", "val", "validation"]),
        ("test", ["test"]),
    ]:
        folders = [
            root / alias
            for alias in aliases
            if (root / alias / "images").is_dir()
            or (root / alias / "_annotations.coco.json").is_file()
        ]
        folders += [
            root / "images" / alias
            for alias in aliases
            if (root / "images" / alias).is_dir()
        ]
        if len(folders) > 1:
            raise ValueError(f"Ambiguous {split} folders")
        if not folders:
            continue
        folder = folders[0]
        coco = folder / "_annotations.coco.json"
        if coco.is_file():
            data = read_json(coco)
            categories = {
                c["id"]: canonical_type(c["name"]) for c in data["categories"]
            }
            if len(categories) != len(data["categories"]) or set(
                categories.values()
            ) != set(TYPES):
                raise ValueError("COCO categories must name the six hold types")
            image_ids = [i["id"] for i in data["images"]]
            annotation_ids = [a["id"] for a in data["annotations"]]
            if len(set(image_ids)) != len(image_ids) or len(set(annotation_ids)) != len(
                annotation_ids
            ):
                raise ValueError("Duplicate COCO image or annotation IDs")
            grouped = {i: [] for i in image_ids}
            for a in data["annotations"]:
                if (
                    a.get("iscrowd")
                    or a["image_id"] not in grouped
                    or a["category_id"] not in categories
                ):
                    raise ValueError("Invalid COCO reference or crowd region")
                x, y, w, h = a["bbox"]
                grouped[a["image_id"]].append(
                    (categories[a["category_id"]], [x, y, x + w, y + h])
                )
            for item in data["images"]:
                path = safe_path(folder, item["file_name"])
                with Image.open(path) as image:
                    if image.size != (item["width"], item["height"]):
                        raise ValueError("COCO dimensions do not match image pixels")
                yield split, path, grouped[item["id"]]
        else:
            names = names or _yolo_names(root, class_names)
            images = folder / "images" if (folder / "images").is_dir() else folder
            labels = (
                folder / "labels" if images != folder else root / "labels" / folder.name
            )
            paths = sorted(
                p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            for path in paths:
                path = safe_path(root, path.relative_to(root))
                label = labels / path.relative_to(images).with_suffix(".txt")
                if not label.is_file():
                    raise ValueError(
                        f"Missing labels: {label}; unlabeled photos cannot train a classifier"
                    )
                with Image.open(path) as image:
                    width, height = image.size
                boxes = []
                for line in label.read_text().splitlines():
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(
                            "Expected YOLO detection labels: class cx cy width height; polygons are unsupported"
                        )
                    index = int(fields[0])
                    cx, cy, w, h = map(float, fields[1:])
                    if index not in range(len(names)):
                        raise ValueError("Unknown YOLO class ID")
                    boxes.append(
                        (
                            names[index],
                            [
                                (cx - w / 2) * width,
                                (cy - h / 2) * height,
                                (cx + w / 2) * width,
                                (cy + h / 2) * height,
                            ],
                        )
                    )
                yield split, path, boxes


def prepare_crops(
    source, output, class_names=None, padding=0.1, groups=None, allow_overlap=False
):
    """Preserve source splits; never randomly split individual crops."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists() or not 0 <= padding <= 0.5:
        raise ValueError(
            "Select an existing labeled export; padding must be in [0, 0.5]"
        )
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Crop output must be outside the source dataset")
    output.mkdir(parents=True, exist_ok=False)
    root = source
    if source.is_file():
        inspect_archive(source)
        root = output / "source"
        with ZipFile(source) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target = root / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
    root = _layout(root)
    group_map = {}
    if groups:
        with Path(groups).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["source_image"] in group_map or not row["group_id"]:
                    raise ValueError(
                        "Group CSV requires unique source_image and nonempty group_id columns"
                    )
                group_map[row["source_image"]] = row["group_id"]
    samples, frames, hashes, group_splits, suspected = [], [], {}, {}, {}
    overlaps = []
    for split, path, annotations in _records(root, class_names):
        relative = path.relative_to(root).as_posix()
        if groups and relative not in group_map:
            raise ValueError(f"Group CSV is missing {relative}")
        group = group_map.get(relative, relative)
        if groups and group in group_splits and group_splits[group] != split:
            raise ValueError(
                f"Group {group} crosses source splits; fix the source split before preparing crops"
            )
        group_splits[group] = split
        with Image.open(path) as original:
            if original.getexif().get(274, 1) != 1:
                raise ValueError(
                    f"Auto-orient images AND annotations before preparation: {relative}"
                )
            image = original.convert("RGB")
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if digest in hashes and hashes[digest][0] != split:
            overlaps.append([hashes[digest][1], relative])
            if not allow_overlap:
                raise ValueError(
                    f"Identical image pixels cross splits: {overlaps[-1]}; fix splits or explicitly --allow-split-overlap"
                )
        hashes[digest] = (split, relative)
        family = re.split(r"\.rf\.", path.stem)[0]
        suspected.setdefault(family, set()).add(split)
        identity = hashlib.sha256(relative.encode()).hexdigest()[:20]
        frames.append(
            {
                "source_image": relative,
                "split": split,
                "group_id": group,
                "pixel_sha256": digest,
                "size_wh": list(image.size),
                "annotations": len(annotations),
            }
        )
        for index, (label, bbox) in enumerate(annotations):
            bbox = _box(bbox, *image.size)
            crop = image.crop(crop_box(bbox, *image.size, padding=padding))
            crop.thumbnail((512, 512), Image.Resampling.LANCZOS)
            name = f"crops/{split}/{identity}_{index:04d}.jpg"
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target, quality=95)
            samples.append(
                {
                    "path": name,
                    "label": TYPES.index(label),
                    "split": split,
                    "source_image": relative,
                    "source_pixel_sha256": digest,
                    "group_id": group,
                    "bbox_xyxy": bbox,
                }
            )
    if not all(any(s["split"] == split for s in samples) for split in ("train", "val")):
        raise ValueError("Nonempty labeled train and validation splits are required")
    data = {
        "schema": "hold-crops-v1",
        "classes": list(TYPES),
        "source": str(source),
        "padding": padding,
        "samples": samples,
        "frames": frames,
        "exact_cross_split_duplicates": overlaps,
        "suspected_cross_split_filename_families": sorted(
            k for k, v in suspected.items() if len(v) > 1
        ),
        "split_protocol": "Source splits preserved; optional group CSV enforced; no crop-level random split",
        "limitations": "Pixel hashes do not detect resized, mirrored or nearby views. Audit wall/scan groups manually.",
        "class_counts": {
            split: {
                name: sum(s["split"] == split and s["label"] == i for s in samples)
                for i, name in enumerate(TYPES)
            }
            for split in ("train", "val", "test")
        },
    }
    (output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")
    return data
