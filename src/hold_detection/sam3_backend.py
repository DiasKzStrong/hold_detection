"""Official SAM 3 text-prompted masks, kept separate from grip classification."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from .assets import sha256
from .geometry import Detection, nms_indices
from .models import resolve_device

SAM3_REPOSITORY = "facebook/sam3"
SAM3_SOURCE_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


def doctor() -> dict:
    import importlib.util
    import torch

    cuda = torch.cuda.is_available()
    return {
        "torch": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "sam3_package_installed": importlib.util.find_spec("sam3") is not None,
        "hf_token_environment_present": bool(os.environ.get("HF_TOKEN")),
        "official_sam3_gpu_runtime_available": cuda,
        "note": "This pinned official SAM 3 implementation allocates CUDA tensors during construction.",
    }


def fetch_checkpoint(destination: Path, revision="main") -> Path:
    # HTTP downloads avoid an unnecessary Xet client and its extra service routes.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Do not change HF_HOME: that would hide tokens saved by `hf auth login`.
    os.environ.setdefault("HF_HUB_CACHE", str(Path.cwd() / ".cache" / "huggingface" / "hub"))
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
    from huggingface_hub.errors import HfHubHTTPError

    try:
        metadata = get_hf_file_metadata(
            hf_hub_url(SAM3_REPOSITORY, "sam3.pt", revision=revision), token=get_token(),
        )
        target = Path(hf_hub_download(
            repo_id=SAM3_REPOSITORY, filename="sam3.pt", revision=metadata.commit_hash,
            local_dir=str(destination),
        ))
    except HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise RuntimeError(
            f"Official SAM 3 checkpoint access failed (HTTP {status}). "
            "Check network access to huggingface.co and cas-bridge.xethub.hf.co, "
            "approved access to facebook/sam3, and HF_TOKEN or hf auth login. "
            "Do not paste tokens into chat."
        ) from None
    except Exception as error:
        # Hub releases use requests, httpx or httpx2. Proxy CONNECT failures can
        # occur before an HTTP response exists; never echo a signed redirect URL.
        if type(error).__module__.split(".")[0] not in {
            "requests", "httpx", "httpx2", "httpcore", "httpcore2",
        }:
            raise
        raise RuntimeError(
            f"Official SAM 3 download failed at the network/proxy layer ({type(error).__name__}). "
            "Allow huggingface.co and cas-bridge.xethub.hf.co in environment settings, "
            "then retry. Approved facebook/sam3 access and Hugging Face authentication "
            "are also required."
        ) from None
    digest = sha256(target)
    expected = metadata.etag.strip('"')
    if re.fullmatch(r"[0-9a-f]{64}", expected) and digest != expected:
        raise ValueError("SAM 3 checksum does not match the official Hub metadata; do not use this file")
    manifest = {
        "repository": SAM3_REPOSITORY, "revision": metadata.commit_hash,
        "filename": "sam3.pt", "sha256": digest, "size_bytes": target.stat().st_size,
        "official_etag": metadata.etag,
    }
    (target.parent / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def decode_predictions(result, width: int, height: int, iou: float, max_det: int):
    """Convert official N×1×H×W masks, keeping identity through filtering/NMS."""
    # Autocast may return BF16, which NumPy cannot represent directly.
    boxes = result["boxes"].detach().float().cpu().numpy()
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    # Official processor masks are already thresholded; keep them compact/binary.
    masks = result["masks"].detach().bool().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("SAM 3 returned masks in unexpected image coordinates")
    if len(boxes) != len(scores) or len(scores) != len(masks):
        raise ValueError("SAM 3 masks, boxes and scores have different counts")
    detections, aligned_masks = [], []
    for box, score, mask in zip(boxes, scores, masks):
        if not np.isfinite(box).all() or not math.isfinite(float(score)):
            continue
        if not 0 <= float(score) <= 1:
            continue
        x1, y1, x2, y2 = map(float, box)
        bbox = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
        mask = np.asarray(mask, dtype=bool)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or not mask.any():
            continue
        detections.append(Detection(bbox, float(score), "hold", label_is_type=False))
        aligned_masks.append((mask, float(score)))
    kept = nms_indices(detections, iou)
    if len(kept) > max_det:
        kept = sorted(sorted(kept, key=lambda i: detections[i].score, reverse=True)[:max_det],
                      key=lambda i: (detections[i].bbox[1], detections[i].bbox[0]))
    return [detections[i] for i in kept], [aligned_masks[i] for i in kept]


class Sam3Detector:
    method = "sam3_text_prompt"
    score_method = "concept_presence_times_instance_score"

    def __init__(self, checkpoint: Path, prompt: str, device="auto", threads=4):
        import torch

        self.device = resolve_device(device)
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError(
                "Official SAM 3 at the pinned revision requires an NVIDIA CUDA GPU; "
                "this runtime has no usable CUDA device. Run holds sam3-doctor. "
                "YOLO prediction and the results gallery still work on CPU."
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing SAM 3 weights: {checkpoint}. Run holds fetch-sam3.")
        manifest = checkpoint.parent / "checkpoint.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text())
            if saved.get("filename") == checkpoint.name and sha256(checkpoint) != saved["sha256"]:
                raise ValueError("Local SAM 3 checkpoint checksum mismatch")
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        torch.set_num_threads(threads)
        # The builder only recognizes 'cuda', not 'cuda:1'; its cache is also device-local.
        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(
                device="cuda", checkpoint_path=str(checkpoint), load_from_HF=False,
                enable_inst_interactivity=False, compile=False,
            )
        self.processor = Sam3Processor(self.model, device=self.device)
        self.prompt = prompt
        self._detections, self._masks = None, None

    def detect(self, image, confidence=0.5, iou=0.5, max_det=500, **kwargs):
        import torch

        if kwargs.get("tile_size", 0):
            raise ValueError("SAM 3 tiled inference is not implemented")
        self._detections, self._masks = None, None
        self.processor.confidence_threshold = confidence
        context = nullcontext()
        if self.device.startswith("cuda"):
            with torch.cuda.device(self.device):
                # Colab T4 can store/emulate BF16 but lacks native BF16 compute.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            context = torch.autocast("cuda", dtype=dtype)
        with torch.inference_mode(), context:
            state = self.processor.set_image(image)
            result = self.processor.set_text_prompt(prompt=self.prompt, state=state)
        self._detections, self._masks = decode_predictions(result, *image.size, iou, max_det)
        return self._detections

    def segment(self, image, detections):
        if self._detections is None or detections != self._detections:
            raise ValueError("SAM 3 masks must be consumed with their corresponding detection call")
        return self._masks
