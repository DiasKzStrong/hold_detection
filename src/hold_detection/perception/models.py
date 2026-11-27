"""Adapters for the reference YOLO/CNN and optional official SAM 2."""

from pathlib import Path
import os

import numpy as np
from PIL import Image

from .geometry import Detection, TYPES, canonical_type, crop_box, merge_detections, tile_windows


def resolve_device(value: str) -> str:
    import torch

    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        return f"cuda:{value}"
    return value


class HoldDetector:
    def __init__(self, weights: Path, device: str, threads: int = 4):
        import torch

        # Keep SDK-generated configuration out of a read-only home directory.
        cache = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".cache" / "ultralytics")))
        (cache / "Ultralytics").mkdir(parents=True, exist_ok=True)
        Path(os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))).mkdir(
            parents=True, exist_ok=True,
        )
        from ultralytics import YOLO

        if not weights.is_file():
            raise FileNotFoundError(f"Missing detector: {weights}. Run: holds fetch-models")
        torch.set_num_threads(threads)
        self.model = YOLO(str(weights), task="detect")
        self.device = resolve_device(device)
        names = self.model.names
        self.labels = {i: canonical_type(names[i]) for i in range(len(names))}
        if set(self.labels.values()) != set(TYPES) or len(names) != len(TYPES):
            raise ValueError("Expected a six-type climbing detector, not a COCO checkpoint")

    def detect(self, image: Image.Image, confidence=0.35, iou=0.5, imgsz=1024,
               tile_size=0, overlap=0.25, max_det=500) -> list[Detection]:
        width, height = image.size
        windows = [(0, 0, width, height)]
        if tile_size and (width > tile_size or height > tile_size):
            windows += tile_windows(width, height, tile_size, overlap)
        detections = []
        for left, top, right, bottom in windows:
            result = self.model.predict(
                image.crop((left, top, right, bottom)), device=self.device,
                conf=confidence, iou=iou, imgsz=imgsz, max_det=max_det,
                agnostic_nms=True, verbose=False,
            )[0]
            if result.boxes is None:
                continue
            for xyxy, score, cls in zip(
                result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                x1, y1, x2, y2 = xyxy
                # Discard clipped objects at INTERNAL tile boundaries, not image edges.
                if ((left > 0 and x1 <= 1) or (top > 0 and y1 <= 1)
                    or (right < width and x2 >= right - left - 1)
                    or (bottom < height and y2 >= bottom - top - 1)):
                    continue
                bbox = (max(0., x1 + left), max(0., y1 + top),
                        min(float(width), x2 + left), min(float(height), y2 + top))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    detections.append(Detection(bbox, score, self.labels[int(cls)]))
        return merge_detections(detections, iou)


class CropClassifier:
    """Checkpoint-compatible implementation of the reference's small two-conv CNN.

    Crop padding, RGB fill and normalization match its process_dataset.py/test_model.py.
    Scores are softmax outputs, not calibrated correctness probabilities.
    """

    def __init__(self, checkpoint: Path, device: str):
        import torch
        from torch import nn
        from torchvision.transforms import Compose, Normalize, ToTensor

        self.device = resolve_device(device)
        layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(), nn.Linear(32 * 32 * 32, 6),
        )
        # Maintain state-dict names without importing or executing upstream scripts.
        self.model = nn.Module()
        self.model.add_module("linear_relu_stack", layers)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.to(self.device).eval()
        self.transform = Compose([
            ToTensor(), Normalize(
                mean=[0.6681055533042859, 0.6137803857704651, 0.5396168694315177],
                std=[0.32921997147304766, 0.3079490945893265, 0.36234998143662933],
            ),
        ])

    def classify_crops(self, crops: list[Image.Image], batch_size=64) -> list[dict[str, float]]:
        import torch

        predictions = []
        for start in range(0, len(crops), batch_size):
            batch = []
            for image in crops[start:start + batch_size]:
                image = image.convert("RGB")
                ratio = 128 / max(image.size)
                resized = image.resize(
                    (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
                    Image.Resampling.BILINEAR,
                )
                canvas = Image.new("RGB", (128, 128), (170, 157, 138))
                canvas.paste(resized, ((128 - resized.width) // 2, (128 - resized.height) // 2))
                batch.append(self.transform(canvas))
            with torch.inference_mode():
                logits = self.model.linear_relu_stack(torch.stack(batch).to(self.device))
                scores = logits.softmax(dim=1).cpu().tolist()
            predictions.extend(dict(zip(TYPES, row)) for row in scores)
        return predictions

    def classify(self, image: Image.Image, detections: list[Detection]):
        return self.classify_crops([
            image.crop(crop_box(d.bbox, *image.size)) for d in detections
        ])


class Sam2Segmenter:
    def __init__(self, checkpoint: Path, config: str, device: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing SAM 2 checkpoint: {checkpoint}")
        self.predictor = SAM2ImagePredictor(
            build_sam2(config, str(checkpoint), device=resolve_device(device))
        )

    def segment(self, image: Image.Image, detections: list[Detection]):
        import torch

        if not detections:
            return []
        output = []
        with torch.inference_mode():
            self.predictor.set_image(np.asarray(image))
            for detection in detections:
                masks, scores, _ = self.predictor.predict(
                    box=np.asarray(detection.bbox, dtype=np.float32), multimask_output=False,
                )
                mask = np.asarray(masks[0], dtype=bool)
                output.append((mask, float(scores[0])))
        return output
