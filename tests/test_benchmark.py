import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from hold_detection.benchmark import compare_runs, detection_matches, rotate_box
from hold_detection.geometry import Detection
from hold_detection.pipeline import run_inference
from hold_detection.visualization import ComparisonViewer, plot_comparison, plot_crop_grid, write_report


class Detector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, **kwargs):
        return self.detections


class Segmenter:
    method = "test_fixture_masks"

    def segment(self, image, detections):
        result = []
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            result.append((mask, 0.9))
        return result


def make_run(tmp_path, name="model", detections=None, masks=False, rotation=0):
    image = tmp_path / "wall.png"
    if not image.exists():
        Image.new("RGB", (30, 20), (100, 120, 140)).save(image)
    if detections is None:
        detections = [Detection((2, 3, 10, 9), 0.9, "jug")]
    output = tmp_path / name
    run_inference(image, output, Detector(detections), segmenter=Segmenter() if masks else None,
                  rotate=rotation, confidence=0.05,
                  model_metadata={"classifier": "detector", "device": "cpu"},
                  infer_grip_types=True)
    return output


def annotations(tmp_path, boxes=None, masks=True, generic=False):
    if boxes is None:
        boxes = [[2, 3, 8, 6]]
    records = []
    for index, (x, y, w, h) in enumerate(boxes):
        item = {"id": index + 1, "image_id": 1, "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0}
        if masks:
            item["segmentation"] = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        records.append(item)
    data = {"images": [{"id": 1, "file_name": "wall.png", "width": 30, "height": 20}],
            "categories": [{"id": 1, "name": "hold" if generic else "jug"}], "annotations": records}
    target = tmp_path / "ground_truth.json"
    target.write_text(json.dumps(data))
    return target


def test_perfect_boxes_and_masks_use_real_coco_metrics(tmp_path):
    run = make_run(tmp_path, masks=True)
    report = compare_runs({"model": run}, {"model": 0.5}, annotations(tmp_path))
    q = report["models"][0]["quality"]
    assert q["precision"] == q["recall"] == q["f1"] == 1
    assert q["box_ap"]["ap50_95"] == pytest.approx(1)
    assert q["mask_ap"]["ap50_95"] == pytest.approx(1)
    assert q["mean_mask_iou_on_box_matched"] == 1
    assert q["classification"]["accuracy_on_matched"] == 1
    assert q["classification"]["end_to_end_correct_type_recall"] == 1


def test_duplicates_are_fp_and_missed_gt_is_not_hidden_by_type_accuracy(tmp_path):
    run = make_run(tmp_path, detections=[Detection((2, 3, 10, 9), 0.9, "jug"),
                                       Detection((2, 3, 10, 9), 0.8, "jug")])
    gt = annotations(tmp_path, boxes=[[2, 3, 8, 6], [16, 3, 8, 6]])
    report = compare_runs({"m": run}, {"m": 0.5}, gt)
    q = report["models"][0]["quality"]
    assert (q["tp"], q["fp"], q["fn"]) == (1, 1, 1)
    assert q["precision"] == q["recall"] == q["f1"] == 0.5
    assert q["classification"]["accuracy_on_matched"] == 1
    assert q["classification"]["typed_gt_detection_coverage"] == 0.5
    assert q["classification"]["end_to_end_correct_type_recall"] == 0.5
    assert q["mask_ap"] is None and q["mask_status"] == "model_has_no_masks"


def test_ap_uses_low_score_candidates_not_operating_point_cutoff(tmp_path):
    run = make_run(tmp_path, detections=[Detection((2, 3, 10, 9), 0.2, "jug")])
    q = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path))["models"][0]["quality"]
    assert q["tp"] == 0 and q["fn"] == 1
    assert q["recall"] == 0 and q["precision"] is None
    assert q["box_ap"]["ap50"] == pytest.approx(1)
    assert q["classification"]["end_to_end_correct_type_recall"] == 0


def test_gt_rotates_with_processed_image_and_masks(tmp_path):
    assert rotate_box([2, 3, 10, 9], 30, 20, 90) == [11, 2, 17, 10]
    run = make_run(tmp_path, masks=True, rotation=90,
                   detections=[Detection((11, 2, 17, 10), 0.9, "jug")])
    q = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path))["models"][0]["quality"]
    assert q["box_ap"]["ap50_95"] == pytest.approx(1)
    assert q["mask_ap"]["ap50_95"] == pytest.approx(1)
    assert q["mean_mask_iou_on_box_matched"] == 1


def test_no_gt_is_unavailable_not_zero_and_generic_gt_has_no_type_metrics(tmp_path):
    run = make_run(tmp_path)
    report = compare_runs({"m": run}, {"m": 0.5})
    assert report["quality_status"] == "ground_truth_required"
    assert report["models"][0]["quality"] is None
    assert report["models"][0]["model_ms_p50"] is not None
    report = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path, generic=True))
    assert report["models"][0]["quality"]["classification"] is None


def test_empty_predictions_are_false_negatives_not_empty_success(tmp_path):
    run = make_run(tmp_path, detections=[])
    q = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path))["models"][0]["quality"]
    assert q["tp"] == q["fp"] == 0 and q["fn"] == 1
    assert q["recall"] == 0 and q["box_ap"]["ap50"] == 0


def test_incomplete_masks_wrong_gt_and_mismatched_comparison_are_rejected(tmp_path):
    run = make_run(tmp_path, masks=True)
    report = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path, masks=False))
    assert report["models"][0]["quality"]["mask_status"] == "ground_truth_masks_incomplete"
    assert report["models"][0]["quality"]["mask_ap"] is None
    with pytest.raises(ValueError, match="candidate floor"):
        compare_runs({"m": run}, {"m": 0.01})
    other = make_run(tmp_path, name="rotated", rotation=90)
    with pytest.raises(ValueError, match="identical"):
        compare_runs({"m": run, "n": other}, {"m": 0.5, "n": 0.5})
    gt = annotations(tmp_path)
    data = json.loads(gt.read_text())
    data["images"][0]["file_name"] = "different.png"
    gt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="match"):
        compare_runs({"m": run}, {"m": 0.5}, gt)


def test_localization_can_be_correct_while_type_is_wrong(tmp_path):
    run = make_run(tmp_path, detections=[Detection((2, 3, 10, 9), 0.9, "crimp")])
    q = compare_runs({"m": run}, {"m": 0.5}, annotations(tmp_path))["models"][0]["quality"]
    assert q["f1"] == 1 and q["classification"]["accuracy_on_matched"] == 0
    assert q["classification"]["end_to_end_correct_type_recall"] == 0


def test_static_report_and_comparison_figures_are_generated(tmp_path):
    one = make_run(tmp_path, name="yolo")
    two = make_run(tmp_path, name="mask_model", masks=True)
    paths = {"YOLO": one, "mask fixture": two}
    report = compare_runs(paths, {"YOLO": 0.5, "mask fixture": 0.5}, annotations(tmp_path))
    output = write_report(paths, report, tmp_path / "report", maximum_frames=1)
    assert (output / "index.html").is_file()
    assert (output / "metrics.csv").is_file()
    assert (output / "quality.png").is_file() and (output / "types.png").is_file()
    with Image.open(output / "frame_000.jpg") as image:
        assert image.width > image.height
    figure = plot_crop_grid(one)
    assert len(figure.axes) == 6
    plt.close(figure)
    with pytest.raises(FileExistsError):
        write_report(paths, report, output)


def test_mask_viewer_composites_one_raster_per_panel(tmp_path):
    run = make_run(tmp_path, masks=True, detections=[Detection((2, 3, 10, 9), 0.9, "jug"),
                                                   Detection((15, 3, 25, 12), 0.8, "crimp")])
    figure = plot_comparison({"mask fixture": run})
    # Hundreds of full-size RGBA artists would exhaust Colab RAM on a busy wall.
    assert all(len(ax.images) == 1 for ax in figure.axes)
    plt.close(figure)


def test_interactive_viewer_uses_current_threshold_and_toggles(tmp_path):
    run = make_run(tmp_path, masks=True)
    viewer = ComparisonViewer({"mask fixture": run}, {"mask fixture": 0.5})
    assert viewer.sliders["mask fixture"].min == 0.05
    viewer.sliders["mask fixture"].value = 0.7
    viewer.masks.value = False
    viewer.boxes.value = False
    viewer.labels.value = True
    assert viewer.thresholds == {"mask fixture": 0.7}
    assert len(plt.get_fignums()) == 0


def test_rle_masks_and_processed_coordinates(tmp_path):
    from pycocotools import mask as mask_utils

    run = make_run(tmp_path, masks=True, rotation=90,
                   detections=[Detection((11, 2, 17, 10), 0.9, "jug")])
    mask = np.zeros((30, 20), dtype=np.uint8, order="F")
    mask[2:10, 11:17] = 1
    rle = mask_utils.encode(mask)
    rle["counts"] = rle["counts"].decode("ascii")
    gt = annotations(tmp_path)
    data = json.loads(gt.read_text())
    data["images"][0].update(width=20, height=30)
    data["annotations"][0].update(bbox=[11, 2, 6, 8], segmentation=rle)
    gt.write_text(json.dumps(data))
    q = compare_runs({"m": run}, {"m": 0.5}, gt, gt_coordinates="processed")["models"][0]["quality"]
    assert q["mask_ap"]["ap50_95"] == pytest.approx(1)
    assert q["mean_mask_iou_on_box_matched"] == 1


def test_unlabeled_template_and_crowd_are_rejected(tmp_path):
    run = make_run(tmp_path)
    gt = annotations(tmp_path)
    data = json.loads(gt.read_text())
    data["annotation_status"] = "unlabeled"
    gt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unlabeled"):
        compare_runs({"m": run}, {"m": 0.5}, gt)
    del data["annotation_status"]
    data["annotations"][0]["iscrowd"] = 1
    gt.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Crowd"):
        compare_runs({"m": run}, {"m": 0.5}, gt)
