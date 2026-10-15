import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest
import torch

from hold_detection import classifier_training as training
from hold_detection.cli import main, parser
from hold_detection.crop_dataset import load_crops, prepare_crops
from hold_detection.geometry import TYPES, Detection
from hold_detection.learned_classifier import LearnedCropClassifier


def labeled_export(root):
    root.mkdir()
    (root / "data.yaml").write_text(
        "names: [Crimp, Jug, Pinch, Pocket, Slope, Volume]\n"
    )
    for split_index, split in enumerate(("train", "valid", "test")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for i in range(12 if split == "train" else 6):
            pixels = np.zeros((40, 48, 3), dtype=np.uint8)
            pixels[:] = [30 + i * 8, 40 + split_index * 45, 80 + i * 3]
            pixels[8:30, 12:35] = [180 - i * 3, 80 + split_index * 25, 40 + i * 7]
            name = f"{split}_{i}"
            Image.fromarray(pixels).save(root / split / "images" / (name + ".png"))
            (root / split / "labels" / (name + ".txt")).write_text(
                f"{i % 6} 0.5 0.5 0.5 0.5\n"
            )
    return root


@pytest.fixture
def crops(tmp_path):
    source = labeled_export(tmp_path / "export")
    output = tmp_path / "crops"
    prepare_crops(source, output)
    return output


def test_yolo_crops_preserve_splits_and_normalize_slope(crops):
    data = load_crops(crops)
    assert data["classes"] == list(TYPES)
    assert [
        sum(r["split"] == s for r in data["samples"]) for s in ["train", "val", "test"]
    ] == [12, 6, 6]
    assert data["class_counts"]["train"]["sloper"] == 2
    assert all(
        len(
            {
                r["split"]
                for r in data["samples"]
                if r["source_image"] == f["source_image"]
            }
        )
        == 1
        for f in data["frames"]
    )


def test_duplicates_and_group_leakage_are_not_silently_accepted(tmp_path):
    root = labeled_export(tmp_path / "export")
    duplicate = root / "valid/images/valid_0.png"
    duplicate.write_bytes((root / "train/images/train_0.png").read_bytes())
    with pytest.raises(ValueError, match="Identical image pixels"):
        prepare_crops(root, tmp_path / "rejected")
    data = prepare_crops(root, tmp_path / "accepted", allow_overlap=True)
    assert len(data["exact_cross_split_duplicates"]) == 1
    groups = tmp_path / "groups.csv"
    groups.write_text(
        "source_image,group_id\n"
        + "\n".join(
            f"{p.relative_to(root).as_posix()},same-wall"
            for p in sorted(root.rglob("*.png"))
        )
    )
    with pytest.raises(ValueError, match="crosses source splits"):
        prepare_crops(root, tmp_path / "group-rejected", groups=groups)


def test_bad_annotations_and_archive_paths_fail(tmp_path):
    root = labeled_export(tmp_path / "export")
    (root / "train/labels/train_0.txt").write_text("0 0.9 0.9 0.8 0.8\n")
    with pytest.raises(ValueError, match="outside the image"):
        prepare_crops(root, tmp_path / "bad")
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("../escape.jpg", b"bad")
    with pytest.raises(ValueError, match="Unsafe archive"):
        prepare_crops(archive, tmp_path / "bad-zip")


def test_coco_export_supports_noncontiguous_category_ids(tmp_path):
    root = tmp_path / "coco"
    for split in ["train", "valid"]:
        folder = root / split
        folder.mkdir(parents=True)
        Image.new("RGB", (40, 30), (20 if split == "train" else 80, 50, 90)).save(
            folder / "wall.jpg"
        )
        (folder / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 7, "file_name": "wall.jpg", "width": 40, "height": 30}
                    ],
                    "categories": [
                        {"id": 10 + i, "name": name} for i, name in enumerate(TYPES)
                    ],
                    "annotations": [
                        {
                            "id": i,
                            "image_id": 7,
                            "category_id": 10 + i,
                            "bbox": [i, 2, 10, 12],
                        }
                        for i in range(6)
                    ],
                }
            )
        )
    data = prepare_crops(root, tmp_path / "prepared")
    assert len(data["samples"]) == 12
    assert [r["label"] for r in data["samples"][:6]] == list(range(6))


def small_resnet(monkeypatch):
    original = training.build_classifier

    def build(*args, **kwargs):
        model = original(*args, **kwargs)
        model.metadata["image_size"] = 32
        return model

    monkeypatch.setattr(training, "build_classifier", build)
    torch.set_num_threads(2)


def test_real_resnet_training_interrupt_resume_and_portable_inference(
    crops, tmp_path, monkeypatch
):
    small_resnet(monkeypatch)
    output = tmp_path / "resnet"
    args = dict(
        architecture="resnet18",
        dataset=crops,
        epochs=1,
        batch_size=6,
        workers=0,
        device="cpu",
        pretrained=False,
        checkpoint_every=1,
    )
    paused = training.train_classifier(output, **args, max_batches_per_run=1)
    assert paused["status"] == "interrupted"
    state = torch.load(output / "last.pt", weights_only=True)
    assert state["next_batch"] == 1 and state["epoch"] == 0
    report = training.train_classifier(output, **args, resume=True)
    assert report["status"] == "completed"
    assert report["metrics"]["test"]["sample_count"] == 6
    assert (output / "learning_curves.png").is_file()
    classifier = LearnedCropClassifier(output / "best.pt", "cpu")
    assert classifier.metadata["initialization"] == "random_test_only"
    result = classifier.classify(
        Image.new("RGB", (40, 30)), [Detection((2, 2, 20, 20), 0.9, "hold", False)]
    )
    assert set(result[0]) == set(TYPES) and sum(result[0].values()) == pytest.approx(1)
    with pytest.raises(ValueError, match="configuration differs"):
        training.train_classifier(output, **{**args, "epochs": 2}, resume=True)
    uninterrupted = tmp_path / "uninterrupted"
    training.train_classifier(uninterrupted, **args)
    expected = torch.load(uninterrupted / "last.pt", weights_only=True)["state_dict"]
    actual = torch.load(output / "last.pt", weights_only=True)["state_dict"]
    assert all(torch.equal(actual[k], v) for k, v in expected.items())


def test_real_siglip_api_cached_features_linear_head_and_offline_checkpoint(
    crops, tmp_path
):
    pytest.importorskip("transformers")
    torch.set_num_threads(2)
    features = tmp_path / "features"
    cache = training.extract_features(
        crops, features, batch_size=6, workers=0, device="cpu", pretrained=False
    )
    assert cache["metadata"]["initialization"] == "random_test_only"
    times = {p: p.stat().st_mtime_ns for p in features.rglob("*.pt")}
    training.extract_features(
        crops, features, batch_size=6, workers=0, device="cpu", pretrained=False
    )
    assert times == {p: p.stat().st_mtime_ns for p in features.rglob("*.pt")}
    output = tmp_path / "siglip"
    result = training.train_classifier(
        output,
        architecture="siglip",
        features=features,
        epochs=2,
        batch_size=6,
        workers=0,
        device="cpu",
        pretrained=False,
    )
    assert result["status"] == "completed"
    classifier = LearnedCropClassifier(output / "best.pt", "cpu")
    values = classifier.classify_crops([Image.new("RGB", (25, 18))])[0]
    assert sum(values.values()) == pytest.approx(1)
    assert classifier.score_method == "siglip_softmax"
    assert (output / "confusion_matrices.png").is_file()
    assert classifier.metadata["vision_config"]["hidden_size"] == 16


def test_multi_gpu_requests_fail_instead_of_silently_using_cpu():
    with pytest.raises(ValueError, match="unavailable"):
        training._parallel(torch.nn.Linear(2, 6), "cpu", 2)


def test_trained_classifier_cli_and_preparation(tmp_path):
    root = labeled_export(tmp_path / "export")
    assert (
        main(["prepare-crops", str(root), "--output", str(tmp_path / "prepared")]) == 0
    )
    args = parser().parse_args(
        [
            "sam3",
            "photos",
            "--output",
            "run",
            "--classifier",
            "trained",
            "--classifier-weights",
            "runs/resnet/best.pt",
        ]
    )
    assert args.classifier == "trained"
