import json
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest

from hold_detection.assets import ASSETS, download_asset
from hold_detection.cli import main, parser
from hold_detection.evaluation import classification_metrics, evaluate_crops
from hold_detection.geometry import Detection, box_iou, canonical_type, crop_box, merge_detections, tile_windows
from hold_detection.pipeline import collect_images, make_hold_record, run_inference
from hold_detection.polycam import GROUPS, inspect_archive, prepare_archive


def test_nms_merges_type_disagreement_but_keeps_hold_on_volume():
    detections = [
        Detection((10, 10, 30, 30), 0.8, "jug"),
        Detection((11, 11, 31, 31), 0.7, "sloper"),
        Detection((0, 0, 100, 100), 0.9, "volume"),
    ]
    merged = merge_detections(detections, 0.5)
    assert {d.label for d in merged} == {"jug", "volume"}
    assert box_iou((1, 1, 1, 1), (1, 1, 1, 1)) == 0


def test_tiles_cover_nondivisible_image_and_crop_bounds():
    windows = tile_windows(1024, 768, 512, 0.25)
    coverage = np.zeros((768, 1024), dtype=bool)
    for left, top, right, bottom in windows:
        coverage[top:bottom, left:right] = True
    assert coverage.all()
    assert len(windows) == len(set(windows))
    assert crop_box((-0.3, 1.1, 30.2, 99.9), 32, 100) == (0, 0, 32, 100)
    with pytest.raises(ValueError):
        tile_windows(10, 10, 5, 1)


def test_unknown_and_volume_do_not_invent_grip_labels():
    low = make_hold_record(Detection((1, 1, 5, 5), 0.3, "jug"), 1, None, 0.5)
    assert low["grip_type"] == "unknown"
    assert low["type_scores"] is None
    assert low["needs_review"] and not low["scores_calibrated"]
    volume = make_hold_record(Detection((1, 1, 5, 5), 0.8, "volume"), 2, None, 0.5)
    assert volume["object_kind"] == "volume" and volume["grip_type"] is None
    assert canonical_type("Slope") == "sloper"
    with pytest.raises(ValueError):
        canonical_type("person")


def test_cnn_disagreement_and_empty_mask_are_reviewed():
    d = Detection((1, 1, 5, 5), 0.8, "jug")
    record = make_hold_record(d, 1, {"crimp": 0.95, "jug": 0.05}, 0.5,
                              (np.zeros((10, 10), dtype=bool), 0.4))
    assert set(record["review_reasons"]) == {"classifier_detector_disagreement", "empty_mask"}
    assert record["predicted_type"] == "crimp"


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, **kwargs):
        return self.detections


def test_pipeline_exports_real_mask_pixels_crops_rotation_and_schema(tmp_path):
    source = tmp_path / "input.png"
    Image.new("RGB", (20, 10), (100, 150, 200)).save(source)
    detection = Detection((2, 3, 8, 14), 0.8, "pinch")

    class Segmenter:
        def segment(self, image, detections):
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[3:14, 2:8] = True
            return [(mask, 0.9)]

    output = tmp_path / "run"
    summary = run_inference(source, output, FakeDetector([detection]),
                            segmenter=Segmenter(), rotate=90)
    assert summary["detection_count"] == 1
    frame_path = next(output.glob("*/holds.json"))
    frame = json.loads(frame_path.read_text())
    assert frame["source_size_wh"] == [20, 10]
    assert frame["processed_size_wh"] == [10, 20]
    assert frame["rotation_clockwise_degrees"] == 90
    record = frame["holds"][0]
    assert record["median_rgb"] == [100, 150, 200]
    assert record["grip_type"] == "pinch"
    mask = np.array(Image.open(frame_path.parent / record["mask_path"]))
    assert np.count_nonzero(mask) == 66
    assert (frame_path.parent / record["crop_path"]).is_file()
    assert Image.open(frame_path.parent / "annotated.jpg").size == (10, 20)
    with pytest.raises(FileExistsError):
        run_inference(source, output, FakeDetector([]))


def test_pipeline_groups_mask_colors_into_routes_and_exports_scene(tmp_path):
    source = tmp_path / "wall.png"
    pixels = np.zeros((12, 30, 3), dtype=np.uint8)
    pixels[:, :10] = (210, 35, 35)
    pixels[:, 10:20] = (190, 45, 45)
    pixels[:, 20:] = (30, 80, 220)
    Image.fromarray(pixels).save(source)
    detections = [
        Detection((0, 0, 10, 12), 0.9, "jug"),
        Detection((10, 0, 20, 12), 0.9, "crimp"),
        Detection((20, 0, 30, 12), 0.9, "sloper"),
    ]

    class Segmenter:
        method = "fixture_masks"

        def segment(self, image, detections):
            masks = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x1, y1, x2, y2 = map(int, detection.bbox)
                mask[y1:y2, x1:x2] = True
                masks.append((mask, 0.95))
            return masks

    output = tmp_path / "run"
    run_inference(source, output, FakeDetector(detections), segmenter=Segmenter())

    scene = json.loads((output / "scene.json").read_text())
    assert scene["schema_version"] == "2.0-draft"
    assert scene["coordinate_space"] == "image"
    assert scene["reconstruction"]["status"] == "not_run"
    assert scene["physical_holds"] == []
    assert len(scene["routes"]) == 2
    assert sorted(route["observation_count"] for route in scene["routes"]) == [1, 2]
    observations = scene["frames"][0]["hold_observations"]
    assert observations[0]["route_id"] == observations[1]["route_id"]
    assert observations[2]["route_id"] != observations[0]["route_id"]
    assert all(observation["color_lab"] for observation in observations)
    assert (next(output.glob("*/routes.jpg"))).is_file()


def test_zero_detections_is_explicit_and_no_output_in_input_tree(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (10, 10)).save(source / "wall.png")
    with pytest.raises(ValueError, match="outside"):
        run_inference(source, source / "output", FakeDetector([]))
    summary = run_inference(source, tmp_path / "empty", FakeDetector([]))
    assert summary["frame_count"] == 1 and summary["detection_count"] == 0
    assert not summary["segmentation_enabled"]


def test_box_only_fallback_exports_reviewable_route_color(tmp_path):
    source = tmp_path / "wall.png"
    Image.new("RGB", (12, 12), (35, 180, 70)).save(source)
    output = tmp_path / "run"
    run_inference(
        source,
        output,
        FakeDetector([Detection((2, 2, 10, 10), 0.8, "jug")]),
    )
    scene = json.loads((output / "scene.json").read_text())
    observation = scene["frames"][0]["hold_observations"][0]
    assert observation["route_id"] == "route_0001"
    assert observation["color_method"] == "bbox_median_rgb_fallback"
    assert observation["needs_review"]
    assert scene["routes"][0]["needs_review"]


def test_image_identity_handles_duplicate_stems(tmp_path):
    source = tmp_path / "images"
    for ext in ("jpg", "png"):
        source.mkdir(exist_ok=True)
        Image.new("RGB", (10, 10)).save(source / f"wall.{ext}")
    summary = run_inference(source, tmp_path / "output", FakeDetector([]))
    assert len({f["frame_id"] for f in summary["frames"]}) == 2


def make_archive(path, extra=None):
    with ZipFile(path, "w") as z:
        for group in GROUPS:
            extension = ".json" if "cameras" in group else ".png" if group in {"depth", "confidence"} else ".jpg"
            z.writestr(f"keyframes/{group}/123{extension}", b"test data")
        if extra:
            z.writestr(extra, b"unsafe")


def test_polycam_import_matches_frames_and_preserves_existing_outputs(tmp_path):
    archive = tmp_path / "raw.zip"
    make_archive(archive)
    assert inspect_archive(archive)["complete_frame_ids"] == ["123"]
    output = tmp_path / "export"
    prepare_archive(archive, output)
    assert (output / "keyframes/depth/123.png").read_bytes() == b"test data"
    assert json.loads((output / "manifest.json").read_text())["groups"]["images"] == 1
    with pytest.raises(FileExistsError):
        prepare_archive(archive, output)
    with pytest.raises(ValueError, match="corrected_images"):
        run_inference(output, tmp_path / "bad", FakeDetector([]))


@pytest.mark.parametrize("name", ["../escape.jpg", "/absolute.jpg", "keyframes/../../escape.jpg"])
def test_polycam_rejects_traversal_before_creating_output(tmp_path, name):
    archive = tmp_path / "raw.zip"
    make_archive(archive, name)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="Unsafe"):
        prepare_archive(archive, output)
    assert not output.exists()


def test_existing_wrong_checkpoint_is_never_overwritten(tmp_path):
    checkpoint = tmp_path / ASSETS["detector"]["filename"]
    checkpoint.write_bytes(b"user checkpoint")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_asset("detector", tmp_path)
    assert checkpoint.read_bytes() == b"user checkpoint"


def test_cli_reports_bad_inputs_without_claiming_success(tmp_path, capsys):
    assert main(["predict", str(tmp_path / "missing"), "--output", str(tmp_path / "run")]) == 1
    assert "No supported images" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    with pytest.raises(SystemExit):
        parser().parse_args(["predict", "wall.jpg", "--output", "run", "--conf", "2"])


def test_metrics_include_minority_class_errors_and_reject_empty_evaluation():
    metrics = classification_metrics([0, 0, 1, 2], [0, 0, 0, 2])
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["jug"]["recall"] == 0
    assert metrics["per_class"]["crimp"]["precision"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix_true_rows"][1][0] == 1
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_evaluation_requires_explicit_valid_labels(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.new("RGB", (128, 128)).save(tmp_path / "images/a.jpg")
    label = tmp_path / "labels/a.txt"
    label.write_text("1 0.5 0.5 0.2 0.2")
    with pytest.raises(ValueError, match="one class ID"):
        evaluate_crops(tmp_path, tmp_path / "report.json", None)

    class Classifier:
        def classify_crops(self, crops):
            return [{"jug": 0.8, "crimp": 0.2} for crop in crops]

    label.write_text("1")
    report = evaluate_crops(tmp_path, tmp_path / "report.json", Classifier())
    assert report["sample_count"] == 1 and report["correct_count"] == 1
