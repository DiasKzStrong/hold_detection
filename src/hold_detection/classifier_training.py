"""Restartable supervised crop training and cached, frozen SigLIP linear probing."""

import csv
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import time

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .assets import sha256
from .crop_dataset import load_crops, manifest_digest, read_json, safe_path
from .evaluation import classification_metrics
from .geometry import TYPES
from .learned_classifier import CropTransform, build_classifier, from_metadata
from .models import resolve_device


def _provenance(device, gpus):
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ["torch", "torchvision", "transformers"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        commit = None
    return {
        "source_commit": commit,
        "packages": packages,
        "device": str(device),
        "gpus_requested": gpus,
        "cuda_version": torch.version.cuda,
        "note": "A Git commit does not describe uncommitted local edits.",
    }


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _cpu_state(model):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _unwrapped(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _parallel(model, device, gpus):
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    if gpus > 1:
        if not str(device).startswith("cuda") or torch.cuda.device_count() < gpus:
            raise ValueError(f"Requested {gpus} CUDA devices but they are unavailable")
        if str(device) not in {"cuda", "cuda:0"}:
            raise ValueError(
                "Multi-GPU mode uses visible CUDA devices starting at cuda:0"
            )
        model.to(device)
        return nn.DataParallel(model, device_ids=list(range(gpus)))
    model.to(device)
    return model


class FeatureEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return self.model.encode(pixels)


class CropDataset(Dataset):
    def __init__(self, root, rows, metadata, training=False, seed=0):
        self.root, self.rows = Path(root), rows
        self.transform = CropTransform(metadata, training)
        self.epoch, self.seed = 0, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_path(self.root, row["path"])) as image:
            # Stateless augmentation permits exact batch-offset restart with workers.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + self.epoch * 1000003 + index)
                pixels = self.transform(image)
        return pixels, row["label"]


def _loader(dataset, batch_size, workers, seed, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )


def _evaluate(model, dataset, device, batch_size, workers=0):
    model.eval()
    truth, predicted, probabilities = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for values, labels in _loader(dataset, batch_size, workers, 0):
            logits = model(values.to(device)).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "Non-finite evaluation logits; no quality report will be published"
                )
            total_loss += nn.functional.cross_entropy(
                logits, labels.to(device), reduction="sum"
            ).item()
            truth.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())
            probabilities.extend(logits.softmax(dim=-1).cpu().tolist())
    metrics = classification_metrics(truth, predicted)
    metrics["loss"] = total_loss / len(truth)
    return metrics, probabilities


def extract_features(
    dataset,
    output,
    batch_size=64,
    workers=4,
    device="auto",
    model_id=None,
    revision="main",
    pretrained=True,
    gpus=1,
):
    """Cache each batch atomically so a preempted encoder pass can be rerun."""
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if batch_size < 1 or workers < 0 or output.is_relative_to(dataset):
        raise ValueError(
            "Use a positive batch size, nonnegative workers and an output outside the dataset"
        )
    manifest = load_crops(dataset)
    fingerprint = manifest_digest(dataset)
    device = resolve_device(device)
    kwargs = {"model_id": model_id} if model_id else {}
    output.mkdir(parents=True, exist_ok=True)
    metadata_file = output / "cache.json"
    if metadata_file.exists():
        cache = read_json(metadata_file)
        if (
            cache.get("schema") != "hold-features-v1"
            or cache["manifest_sha256"] != fingerprint
            or cache["batch_size"] != batch_size
            or cache["pretrained"] != pretrained
            or (model_id and cache["metadata"]["model_id"] != model_id)
            or cache["requested_revision"] != revision
        ):
            raise ValueError(
                "Feature cache configuration differs; choose a new output directory"
            )
        model = from_metadata(cache["metadata"])
        model.encoder.load_state_dict(
            torch.load(output / "encoder.pt", map_location="cpu", weights_only=True)
        )
    else:
        if any(output.iterdir()):
            raise ValueError(
                "Feature output is not an initialized cache; choose a new directory"
            )
        model = build_classifier(
            "siglip", pretrained=pretrained, revision=revision, **kwargs
        )
        model.metadata["crop_padding"] = manifest["padding"]
        atomic_save(_cpu_state(model.encoder), output / "encoder.pt")
        cache = {
            "schema": "hold-features-v1",
            "manifest_sha256": fingerprint,
            "metadata": model.metadata,
            "encoder_sha256": sha256(output / "encoder.pt"),
            "batch_size": batch_size,
            "pretrained": pretrained,
            "requested_revision": revision,
            "provenance": _provenance(device, gpus),
            "split_protocol": manifest["split_protocol"],
            "rows": manifest["samples"],
        }
        metadata_file.write_text(json.dumps(cache, indent=2) + "\n")
    if sha256(output / "encoder.pt") != cache["encoder_sha256"]:
        raise ValueError("Cached encoder checksum mismatch")
    model.eval().requires_grad_(False)
    encoder = _parallel(FeatureEncoder(model), device, gpus).eval()
    for split in ("train", "val", "test"):
        rows = [r for r in manifest["samples"] if r["split"] == split]
        if not rows:
            continue
        data = CropDataset(dataset, rows, model.metadata)
        pending = []
        for index in range(math.ceil(len(rows) / batch_size)):
            shard = output / split / f"{index:06d}.pt"
            if shard.exists():
                saved = torch.load(shard, map_location="cpu", weights_only=True)
                labels = torch.tensor(
                    [
                        r["label"]
                        for r in rows[index * batch_size : (index + 1) * batch_size]
                    ]
                )
                if not torch.equal(saved["labels"], labels) or saved[
                    "features"
                ].shape != (len(labels), model.metadata["feature_dim"]):
                    raise ValueError("Invalid cached feature shard")
                if not torch.isfinite(saved["features"]).all():
                    raise ValueError("Cached features contain non-finite values")
                continue
            pending.append(index)
        batches = [
            list(range(i * batch_size, min((i + 1) * batch_size, len(rows))))
            for i in pending
        ]
        loader = DataLoader(
            data,
            batch_sampler=batches,
            num_workers=workers,
            generator=torch.Generator().manual_seed(0),
            pin_memory=str(device).startswith("cuda"),
        )
        for index, (pixels, labels) in zip(pending, loader):
            shard = output / split / f"{index:06d}.pt"
            with torch.inference_mode():
                features = encoder(pixels.to(device)).float().cpu()
            shard.parent.mkdir(parents=True, exist_ok=True)
            atomic_save({"features": features, "labels": labels}, shard)
            print(
                f"Features {split}: {min((index + 1) * batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )
    # A completion marker is emitted only after all three source splits are covered.
    (output / "complete.json").write_text(
        json.dumps({"cache_sha256": sha256(metadata_file)}) + "\n"
    )
    return cache


def _feature_data(output):
    output = Path(output).resolve()
    cache = read_json(output / "cache.json")
    complete = read_json(output / "complete.json")
    if complete["cache_sha256"] != sha256(output / "cache.json"):
        raise ValueError("Feature cache is incomplete or changed")
    if sha256(output / "encoder.pt") != cache["encoder_sha256"]:
        raise ValueError("Cached encoder checksum mismatch")
    result = {}
    for split in ("train", "val", "test"):
        rows = [r for r in cache["rows"] if r["split"] == split]
        if not rows:
            continue
        shards = [
            torch.load(p, map_location="cpu", weights_only=True)
            for p in sorted((output / split).glob("*.pt"))
        ]
        features = torch.cat([s["features"] for s in shards])
        labels = torch.cat([s["labels"] for s in shards]).long()
        if labels.tolist() != [r["label"] for r in rows] or features.shape != (
            len(rows),
            cache["metadata"]["feature_dim"],
        ):
            raise ValueError("Cached features do not match source sample order")
        if not torch.isfinite(features).all():
            raise ValueError("Cached features contain non-finite values")
        result[split] = TensorDataset(features, labels)
    return cache, result


def _plots(output, history, metrics):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        [r["epoch"] for r in history], [r["train_loss"] for r in history], label="train"
    )
    axes[0].plot(
        [r["epoch"] for r in history],
        [r["val_loss"] for r in history],
        label="validation",
    )
    axes[0].set(xlabel="Epoch", ylabel="Cross entropy", title="Training history")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in history], [r["val_macro_f1"] for r in history])
    axes[1].set(
        xlabel="Epoch",
        ylabel="Macro-F1 (all six classes)",
        ylim=(0, 1),
        title="Validation, not test",
    )
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=140)
    plt.close(figure)
    evaluated = [(k, v) for k, v in metrics.items() if k in {"validation", "test"}]
    figure, axes = plt.subplots(
        1, len(evaluated), figsize=(6 * len(evaluated), 5), squeeze=False
    )
    for ax, (name, values) in zip(axes[0], evaluated):
        matrix = values["confusion_matrix_true_rows"]
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(6), TYPES, rotation=45, ha="right")
        ax.set_yticks(range(6), TYPES)
        ax.set(title=name, xlabel="Predicted", ylabel="True")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrices.png", dpi=140)
    plt.close(figure)


def train_classifier(
    output,
    architecture="resnet18",
    dataset=None,
    features=None,
    epochs=20,
    batch_size=64,
    workers=4,
    learning_rate=None,
    seed=0,
    device="auto",
    resume=False,
    checkpoint_every=100,
    time_budget_seconds=0,
    max_batches_per_run=0,
    pretrained=True,
    gpus=1,
):
    """Best epoch uses validation macro-F1. Test is evaluated only after training."""
    if (
        epochs < 1
        or batch_size < 1
        or workers < 0
        or checkpoint_every < 1
        or time_budget_seconds < 0
        or not math.isfinite(time_budget_seconds)
        or max_batches_per_run < 0
    ):
        raise ValueError("Invalid training configuration")
    torch.manual_seed(seed)
    random.seed(seed)
    device = resolve_device(device)
    provenance = _provenance(device, gpus)
    linear = architecture == "siglip"
    learning_rate = (
        (0.01 if linear else 0.0001) if learning_rate is None else learning_rate
    )
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Learning rate must be positive and finite")
    output = Path(output).resolve()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output / "last.pt").is_file():
        raise ValueError("--resume requires an existing output/last.pt")
    restored = (
        torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if resume
        else None
    )
    if linear:
        if not features:
            raise ValueError(
                "SigLIP linear probing requires --features from extract-features"
            )
        cache, data = _feature_data(features)
        metadata = cache["metadata"]
        fingerprint = cache["manifest_sha256"] + ":" + cache["encoder_sha256"]
        model = nn.Linear(metadata["feature_dim"], len(TYPES))
    else:
        if architecture != "resnet18" or not dataset:
            raise ValueError("ResNet18 training requires --dataset with prepared crops")
        manifest = load_crops(dataset)
        fingerprint = manifest_digest(dataset)
        model = (
            from_metadata(restored["metadata"])
            if restored
            else build_classifier("resnet18", pretrained=pretrained)
        )
        metadata = model.metadata
        metadata["crop_padding"] = manifest["padding"]
        data = {
            split: CropDataset(
                dataset,
                [r for r in manifest["samples"] if r["split"] == split],
                metadata,
                split == "train",
                seed,
            )
            for split in ("train", "val", "test")
        }
        data = {k: v for k, v in data.items() if len(v)}
    if not all(k in data for k in ("train", "val")):
        raise ValueError("Nonempty train and validation data are required")
    targets = (
        data["train"].tensors[1].tolist()
        if linear
        else [r["label"] for r in data["train"].rows]
    )
    counts = torch.bincount(torch.tensor(targets), minlength=6)
    if not bool((counts > 0).all()):
        raise ValueError("Training data must include every hold class")
    config = {
        "architecture": architecture,
        "fingerprint": fingerprint,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "pretrained": pretrained,
    }
    if restored and restored["config"] != config:
        raise ValueError(
            "Resume configuration differs from last.pt; use the original training configuration"
        )
    model = _parallel(model, device, gpus)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = (counts.sum() / (6 * counts.float())).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    history, best_score, start_epoch, next_batch = [], -1.0, 0, 0
    accum = [0.0, 0]
    if restored:
        _unwrapped(model).load_state_dict(restored["state_dict"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        history, best_score = restored["history"], restored["best_score"]
        start_epoch, next_batch, accum = (
            restored["epoch"],
            restored["next_batch"],
            restored["accum"],
        )
        torch.set_rng_state(restored["rng"])
    handlers, stop = {}, {"requested": False}

    def request_stop(*_):
        stop["requested"] = True

    for name in ("SIGTERM", "SIGUSR1"):
        sig = getattr(signal, name, None)
        if sig is not None:
            handlers[sig] = signal.signal(sig, request_stop)
    started, batches = time.monotonic(), 0

    def checkpoint(epoch, offset, running):
        atomic_save(
            {
                "schema": "hold-training-v1",
                "provenance": provenance,
                "config": config,
                "metadata": metadata,
                "state_dict": _cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_batch": offset,
                "accum": running,
                "history": history,
                "best_score": best_score,
                "rng": torch.get_rng_state(),
            },
            output / "last.pt",
        )

    try:
        for epoch in range(start_epoch, epochs):
            if not linear:
                data["train"].epoch = epoch
            model.train()
            loader = _loader(data["train"], batch_size, workers, seed + epoch, True)
            running = list(accum) if epoch == start_epoch else [0.0, 0]
            for index, (values, labels) in enumerate(loader):
                if epoch == start_epoch and index < next_batch:
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(values.to(device)).float()
                loss = criterion(logits, labels.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                running[0] += loss.item() * len(labels)
                running[1] += len(labels)
                batches += 1
                interrupted = (
                    stop["requested"]
                    or (
                        time_budget_seconds
                        and time.monotonic() - started >= time_budget_seconds
                    )
                    or (max_batches_per_run and batches >= max_batches_per_run)
                )
                if batches % checkpoint_every == 0 or interrupted:
                    checkpoint(epoch, index + 1, running)
                if interrupted:
                    print(
                        f"Checkpoint saved: {output / 'last.pt'}; rerun with --resume",
                        flush=True,
                    )
                    return {"status": "interrupted", "output": str(output)}
            val, _ = _evaluate(model, data["val"], device, batch_size, workers)
            score = val["macro_f1_all_six_classes"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": running[0] / running[1],
                    "val_loss": val["loss"],
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": score,
                }
            )
            if score > best_score:
                best_score = score
                atomic_save(
                    {
                        "schema": "hold-classifier-v1",
                        "provenance": provenance,
                        "metadata": metadata,
                        "state_dict": _cpu_state(model),
                        "selected_epoch": epoch + 1,
                        "config": config,
                    },
                    output / ("best_head.pt" if linear else "best.pt"),
                )
            scheduler.step()
            checkpoint(epoch + 1, 0, [0.0, 0])
            next_batch = 0
            with (output / "history.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            print(json.dumps(history[-1]), flush=True)
        best = torch.load(
            output / ("best_head.pt" if linear else "best.pt"),
            map_location="cpu",
            weights_only=True,
        )
        _unwrapped(model).load_state_dict(best["state_dict"])
        metrics = {
            "schema": "hold-training-report-v1",
            "provenance": provenance,
            "architecture": architecture,
            "selection": "validation macro-F1",
            "selected_epoch": best["selected_epoch"],
            "classes": list(TYPES),
            "config": config,
            "limitations": "Crop classification only. No claim of gym generalization, detection recall or score calibration.",
        }
        for split, name in [("val", "validation"), ("test", "test")]:
            if split in data:
                metrics[name], probabilities = _evaluate(
                    model, data[split], device, batch_size, workers
                )
                rows = (
                    [r for r in cache["rows"] if r["split"] == split]
                    if linear
                    else data[split].rows
                )
                with (output / f"{split}_predictions.jsonl").open("w") as stream:
                    for row, values in zip(rows, probabilities):
                        stream.write(
                            json.dumps(
                                {
                                    "path": row["path"],
                                    "source_image": row["source_image"],
                                    "truth": TYPES[row["label"]],
                                    "scores": dict(zip(TYPES, values)),
                                }
                            )
                            + "\n"
                        )
        if linear:
            encoder = torch.load(
                Path(features) / "encoder.pt", map_location="cpu", weights_only=True
            )
            if sha256(Path(features) / "encoder.pt") != cache["encoder_sha256"]:
                raise ValueError("Feature encoder checksum mismatch")
            best["state_dict"] = {
                **{"encoder." + k: v for k, v in encoder.items()},
                **{"head." + k: v for k, v in best["state_dict"].items()},
            }
            atomic_save(best, output / "best.pt")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        _plots(output, history, metrics)
        return {
            "status": "completed",
            "output": str(output),
            "checkpoint": str(output / "best.pt"),
            "metrics": metrics,
        }
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
