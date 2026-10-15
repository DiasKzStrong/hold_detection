"""Portable ResNet18 and frozen-SigLIP crop classifiers (no Hub access at inference)."""

import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from PIL import Image

from .geometry import TYPES, crop_box
from .models import resolve_device

SIGLIP_MODEL = "google/siglip-base-patch16-224"


class CropTransform:
    def __init__(self, metadata, training=False):
        self.size = metadata["image_size"]
        self.fill = tuple(round(v * 255) for v in metadata["mean"])
        self.augment = (
            transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.15, contrast=0.15, saturation=0.1
                    ),
                ]
            )
            if training
            else None
        )
        self.finish = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(metadata["mean"], metadata["std"]),
            ]
        )

    def __call__(self, image):
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        ratio = self.size / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.BICUBIC,
        )
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(
            image, ((self.size - image.width) // 2, (self.size - image.height) // 2)
        )
        if self.augment:
            canvas = self.augment(canvas)
        return self.finish(canvas)


class HoldClassifier(nn.Module):
    def __init__(self, encoder, metadata):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(metadata["feature_dim"], len(TYPES))
        self.metadata = metadata

    def encode(self, pixels):
        if self.metadata["architecture"] == "siglip":
            features = self.encoder(pixel_values=pixels).pooler_output
            return F.normalize(features.float(), dim=-1)
        return self.encoder(pixels)

    def forward(self, pixels):
        return self.head(self.encode(pixels))


def build_classifier(
    architecture, pretrained=True, model_id=SIGLIP_MODEL, revision="main"
):
    if architecture == "resnet18":
        encoder = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        encoder.fc = nn.Identity()
        metadata = {
            "architecture": architecture,
            "feature_dim": 512,
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "initialization": "ImageNet1K_V1" if pretrained else "random_test_only",
        }
    elif architecture == "siglip":
        from transformers import (
            SiglipImageProcessor,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        if pretrained:
            from huggingface_hub import hf_hub_download

            local = Path(model_id).is_dir()
            offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {
                "1",
                "true",
                "yes",
            }
            if not local:
                config = hf_hub_download(
                    model_id, "config.json", revision=revision, local_files_only=offline
                )
                revision = Path(config).parent.name
            kwargs = (
                {} if local else {"revision": revision, "local_files_only": offline}
            )
            encoder = SiglipVisionModel.from_pretrained(
                model_id, use_safetensors=True, **kwargs
            )
            processor = SiglipImageProcessor.from_pretrained(model_id, **kwargs)
            mean, std = list(processor.image_mean), list(processor.image_std)
        else:
            # Tiny offline encoder exercises the real SigLIP API in smoke tests only.
            encoder = SiglipVisionModel(
                SiglipVisionConfig(
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    image_size=32,
                    patch_size=16,
                )
            )
            mean, std = [0.5] * 3, [0.5] * 3
        metadata = {
            "architecture": architecture,
            "feature_dim": encoder.config.hidden_size,
            "image_size": encoder.config.image_size,
            "mean": mean,
            "std": std,
            "model_id": model_id,
            "revision": revision if pretrained else None,
            "vision_config": encoder.config.to_dict(),
            "initialization": "pretrained" if pretrained else "random_test_only",
        }
        encoder.requires_grad_(False).eval()
    else:
        raise ValueError("Architecture must be resnet18 or siglip")
    metadata.update(
        classes=list(TYPES),
        preprocessing="RGB; aspect-preserving letterbox; bicubic; normalization",
        crop_padding=0.1,
        scores_calibrated=False,
    )
    return HoldClassifier(encoder, metadata)


def from_metadata(metadata):
    if metadata.get("classes") != list(TYPES):
        raise ValueError("Classifier checkpoint class order is incompatible")
    if metadata["architecture"] == "resnet18":
        encoder = models.resnet18(weights=None)
        encoder.fc = nn.Identity()
    elif metadata["architecture"] == "siglip":
        from transformers import SiglipVisionConfig, SiglipVisionModel

        encoder = SiglipVisionModel(
            SiglipVisionConfig.from_dict(metadata["vision_config"])
        )
        encoder.requires_grad_(False).eval()
    else:
        raise ValueError("Unsupported classifier checkpoint architecture")
    return HoldClassifier(encoder, metadata)


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("schema") != "hold-classifier-v1":
        raise ValueError("Expected a trained hold-classifier-v1 best.pt artifact")
    return data


class LearnedCropClassifier:
    def __init__(self, checkpoint, device="auto"):
        data = load_checkpoint(checkpoint)
        self.metadata = data["metadata"]
        self.device = resolve_device(device)
        self.model = from_metadata(self.metadata)
        self.model.load_state_dict(data["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.transform = CropTransform(self.metadata)
        self.score_method = f"{self.metadata['architecture']}_softmax"

    def classify_crops(self, crops, batch_size=32):
        scores = []
        with torch.inference_mode():
            for start in range(0, len(crops), batch_size):
                pixels = torch.stack(
                    [self.transform(c) for c in crops[start : start + batch_size]]
                ).to(self.device)
                values = self.model(pixels).float().softmax(dim=-1).cpu().tolist()
                scores.extend(dict(zip(TYPES, row)) for row in values)
        return scores

    def classify(self, image, detections):
        padding = self.metadata["crop_padding"]
        return self.classify_crops(
            [
                image.crop(crop_box(d.bbox, *image.size, padding=padding))
                for d in detections
            ]
        )
