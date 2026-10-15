import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from hold_detection.assets import sha256
from hold_detection.cli import main, parser
from hold_detection.gallery import asset_url, build_gallery
from hold_detection.geometry import Detection
from hold_detection.pipeline import make_hold_record, run_inference
from hold_detection.sam3_backend import Sam3Detector, decode_predictions, fetch_checkpoint


def predictions():
    masks = torch.zeros((4, 1, 20, 30), dtype=torch.bool)
    masks[0, 0, 8:18, 10:20] = True
    masks[1, 0, 1:5, 1:5] = True
    masks[2, 0, 8:18, 11:21] = True
    # The fourth is an empty mask and must not become a hold.
    return {"boxes": torch.tensor([[10, 8, 20, 18], [1, 1, 5, 5], [11, 8, 21, 18], [1, 8, 8, 15]]),
            "scores": torch.tensor([0.8, 0.9, 0.7, 0.95]), "masks": masks}


def test_sam3_nms_preserves_correct_masks_and_does_not_invent_types():
    detections, masks = decode_predictions(predictions(), 30, 20, 0.5, 500)
    assert len(detections) == 2
    assert detections[0].bbox == (1, 1, 5, 5)
    assert masks[0][0].sum() == 16
    assert masks[1][0].sum() == 100
    record = make_hold_record(detections[0], 1, None, 0.5, masks[0], "sam3_text_prompt")
    assert record["object_kind"] == "hold" and record["grip_type"] == "unknown"
    assert record["type_score"] is None
    assert record["review_reasons"] == ["type_not_inferred"]
    classified = make_hold_record(detections[0], 1, {"jug": 0.9, "crimp": 0.1}, 0.5)
    assert classified["grip_type"] == "jug" and not classified["needs_review"]


def test_sam3_adapter_consumes_official_processor_outputs_and_rejects_stale_masks(tmp_path):
    class Processor:
        def set_image(self, image):
            assert image.size == (30, 20)
            return {"image_set": True}

        def set_text_prompt(self, prompt, state):
            assert state["image_set"] and prompt == "climbing hold"
            return predictions()

    detector = Sam3Detector.__new__(Sam3Detector)
    detector.device, detector.prompt = "cpu", "climbing hold"
    detector.processor = Processor()
    detector._detections, detector._masks = None, None
    with pytest.raises(ValueError, match="corresponding"):
        detector.segment(None, [])
    source = tmp_path / "wall.jpg"
    Image.new("RGB", (30, 20), (60, 90, 120)).save(source)
    output = tmp_path / "run"
    summary = run_inference(source, output, detector, segmenter=detector)
    assert summary["segmentation_enabled"] and summary["detection_count"] == 2
    frame_path = next(output.glob("*/holds.json"))
    frame = json.loads(frame_path.read_text())
    assert frame["holds"][0]["segmentation_method"] == "sam3_text_prompt"
    assert frame["holds"][0]["segmentation_score_method"] == "concept_presence_times_instance_score"
    assert all(h["type_score"] is None for h in frame["holds"])
    html = (output / "index.html").read_text()
    assert "mask PNG" in html and "type not inferred" in html


def test_sam3_checkpoint_download_pins_resolved_commit_and_checks_official_hash(tmp_path, monkeypatch):
    weight = tmp_path / "sam3.pt"
    weight.write_bytes(b"test checkpoint bytes")
    calls = []

    class HubError(Exception):
        pass

    def download(**kwargs):
        calls.append(kwargs)
        return str(weight)

    metadata = SimpleNamespace(commit_hash="a" * 40, etag=sha256(weight))
    fake_hub = SimpleNamespace(
        get_hf_file_metadata=lambda *a, **k: metadata,
        get_token=lambda: None,
        hf_hub_download=download,
        hf_hub_url=lambda *a, **k: "https://huggingface.co/facebook/sam3/resolve/main/sam3.pt",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", SimpleNamespace(HfHubHTTPError=HubError))
    assert fetch_checkpoint(tmp_path) == weight
    assert calls[0]["revision"] == "a" * 40
    assert json.loads((tmp_path / "checkpoint.json").read_text())["sha256"] == sha256(weight)
    metadata.etag = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        fetch_checkpoint(tmp_path)


def test_sam3_fails_clearly_without_cuda_before_creating_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = tmp_path / "wall.jpg"
    Image.new("RGB", (30, 20)).save(image)
    assert main(["sam3", str(image), "--output", str(tmp_path / "run")]) == 1
    assert "requires an NVIDIA CUDA GPU" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()
    args = parser().parse_args(["sam3", str(image), "--output", "run"])
    assert args.prompt == "climbing hold" and args.classifier == "cnn"


def test_gallery_escapes_labels_and_rejects_external_assets_and_user_files(tmp_path):
    run = tmp_path / "run"
    frame = run / "frame"
    frame.mkdir(parents=True)
    record = make_hold_record(Detection((0, 0, 5, 5), 0.8, "jug"), 1, None, 0.5)
    record["crop_path"] = "crop.jpg"
    (run / "summary.json").write_text(json.dumps({"frame_count": 1, "detection_count": 1,
                                                "segmentation_enabled": False}))
    (frame / "holds.json").write_text(json.dumps({"source_image": '<script>alert(1)</script>.jpg', "holds": [record]}))
    path = build_gallery(run)
    html = path.read_text()
    assert "<script>alert(1)</script>" not in html
    assert "script&gt;.jpg" in html
    assert build_gallery(run) == path
    with pytest.raises(ValueError, match="escapes"):
        asset_url(run, frame, "../../private.jpg")
    path.write_text("user-written page")
    with pytest.raises(FileExistsError):
        build_gallery(run)
