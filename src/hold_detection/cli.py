"""Command line entry points; imports of large models are deferred."""

import argparse
import json
from pathlib import Path
import sys
from urllib.error import URLError


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Expected a number in [0, 1]")
    return value


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def parser():
    root = argparse.ArgumentParser(description="DeMoSearch climbing hold perception")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser(
        "fetch-models", help="Download verified reference checkpoints"
    )
    fetch.add_argument("--output", type=Path, default=Path("models"))
    fetch.add_argument(
        "--model", choices=("detector", "classifier", "all"), default="all"
    )
    sam_fetch = commands.add_parser(
        "fetch-sam3", help="Download gated official facebook/sam3 weights"
    )
    sam_fetch.add_argument("--output", type=Path, default=Path("models/sam3"))
    sam_fetch.add_argument("--revision", default="main")
    commands.add_parser(
        "sam3-doctor", help="Report SAM 3 software, GPU and auth prerequisites"
    )
    encoders = commands.add_parser(
        "fetch-encoders", help="Warm pretrained ResNet18/SigLIP caches before GPU jobs"
    )
    encoders.add_argument(
        "--architecture", choices=("resnet18", "siglip", "all"), default="all"
    )
    encoders.add_argument("--revision", default="main")
    crops = commands.add_parser(
        "prepare-crops", help="Prepare source-split-preserving YOLO/COCO training crops"
    )
    crops.add_argument("source", type=Path)
    crops.add_argument("--output", type=Path, required=True)
    crops.add_argument("--class-names", nargs=6)
    crops.add_argument("--padding", type=probability, default=0.1)
    crops.add_argument(
        "--groups",
        type=Path,
        help="CSV: source_image,group_id relative to labeled export root",
    )
    crops.add_argument(
        "--allow-split-overlap",
        action="store_true",
        help="Explicitly accept reported exact train/eval duplicates",
    )
    features = commands.add_parser(
        "extract-features",
        help="Cache frozen SigLIP embeddings; reruns resume completed shards",
    )
    features.add_argument("--dataset", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--model-id", default="google/siglip-base-patch16-224")
    features.add_argument("--revision", default="main")
    features.add_argument("--batch-size", type=positive, default=64)
    features.add_argument("--workers", type=int, default=4)
    features.add_argument("--gpus", type=positive, default=1)
    features.add_argument("--device", default="auto")
    features.add_argument("--threads", type=positive, default=4)
    features.add_argument(
        "--random-init",
        action="store_true",
        help="Tiny random SigLIP for offline smoke tests ONLY",
    )
    train = commands.add_parser(
        "train-classifier",
        help="Fine-tune ResNet18 or train a frozen-SigLIP linear head",
    )
    train.add_argument("--architecture", choices=("resnet18", "siglip"), required=True)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--features", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=positive, default=20)
    train.add_argument("--batch-size", type=positive, default=64)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--gpus", type=positive, default=1)
    train.add_argument("--device", default="auto")
    train.add_argument("--threads", type=positive, default=4)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--checkpoint-every", type=positive, default=100)
    train.add_argument("--time-budget-seconds", type=float, default=0)
    train.add_argument(
        "--max-batches-per-run",
        type=int,
        default=0,
        help="Operational checkpoint/stop budget; zero disables",
    )
    train.add_argument(
        "--random-init",
        action="store_true",
        help="Disable ImageNet initialization for offline smoke tests ONLY",
    )
    gallery = commands.add_parser(
        "gallery", help="Build an offline HTML gallery for a completed run"
    )
    gallery.add_argument("run", type=Path)
    benchmark = commands.add_parser(
        "benchmark", help="Compare shared-image runs; optional COCO quality metrics"
    )
    benchmark.add_argument(
        "--run", nargs=2, action="append", metavar=("MODEL", "PATH"), required=True
    )
    benchmark.add_argument(
        "--threshold",
        nargs=2,
        action="append",
        metavar=("MODEL", "SCORE"),
        required=True,
    )
    benchmark.add_argument("--annotations", type=Path)
    benchmark.add_argument(
        "--gt-coordinates", choices=("source", "processed"), default="source"
    )
    benchmark.add_argument("--match-iou", type=probability, default=0.5)
    benchmark.add_argument("--max-det", type=positive, default=500)
    benchmark.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect-polycam", help="Inspect a raw-data ZIP without extracting"
    )
    inspect.add_argument("archive", type=Path)
    prepare = commands.add_parser(
        "prepare-polycam", help="Unpack RGB-D frames and camera metadata"
    )
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    inspect_wall = commands.add_parser(
        "inspect-wall-bundle", help="Validate one project wall folder"
    )
    inspect_wall.add_argument("directory", type=Path)
    prepare_wall = commands.add_parser(
        "prepare-wall-bundle", help="Prepare one wall's RGB-D capture and GLB"
    )
    prepare_wall.add_argument("directory", type=Path)
    prepare_wall.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate-crops", help="Evaluate the CNN on labeled crops"
    )
    evaluate.add_argument(
        "dataset", type=Path, help="Directory with images/ and labels/<stem>.txt"
    )
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--weights", type=Path, default=Path("models/climbing-cnn.pth")
    )
    evaluate.add_argument("--classifier", choices=("cnn", "trained"), default="cnn")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--threads", type=positive, default=4)
    predict = commands.add_parser(
        "predict", help="Find holds with the legacy YOLO detector; optionally refine masks"
    )
    predict.add_argument(
        "source", type=Path, help="An image or directory of RGB images"
    )
    predict.add_argument(
        "--output", type=Path, required=True, help="A NEW run directory"
    )
    predict.add_argument(
        "--weights", type=Path, default=Path("models/climbing-yolo11s.pt")
    )
    predict.add_argument(
        "--classifier", choices=("none", "detector", "cnn", "trained"), default="none"
    )
    predict.add_argument(
        "--classifier-weights", type=Path, default=Path("models/climbing-cnn.pth")
    )
    predict.add_argument("--device", default="auto")
    predict.add_argument("--threads", type=positive, default=4)
    predict.add_argument("--conf", type=probability, default=0.35)
    predict.add_argument("--min-type-score", type=probability, default=0.5)
    predict.add_argument("--iou", type=probability, default=0.5)
    predict.add_argument("--imgsz", type=positive, default=1024)
    predict.add_argument(
        "--tile-size", type=positive, help="Also detect on overlapping image tiles"
    )
    predict.add_argument("--overlap", type=probability, default=0.25)
    predict.add_argument("--max-det", type=positive, default=500)
    predict.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Clockwise rotation before detection; output uses rotated coordinates",
    )
    predict.add_argument("--limit", type=positive)
    predict.add_argument("--sam2-checkpoint", type=Path)
    predict.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    sam = commands.add_parser(
        "sam3", help="Find holds with text-prompted SAM 3 masks; optionally classify"
    )
    sam.add_argument("source", type=Path)
    sam.add_argument("--output", type=Path, required=True)
    sam.add_argument("--weights", type=Path, default=Path("models/sam3/sam3.pt"))
    sam.add_argument(
        "--prompt", action="append",
        help="Repeatable hold prompt; defaults target holds, grips, and footholds",
    )
    sam.add_argument(
        "--volume-prompt", action="append",
        help="Repeatable Volume prompt; defaults to 'climbing volume'",
    )
    sam.add_argument("--classifier", choices=("none", "cnn", "trained"), default="none")
    sam.add_argument(
        "--classifier-weights", type=Path, default=Path("models/climbing-cnn.pth")
    )
    sam.add_argument("--device", default="auto")
    sam.add_argument("--threads", type=positive, default=4)
    sam.add_argument("--conf", type=probability, default=0.5)
    sam.add_argument("--min-type-score", type=probability, default=0.5)
    sam.add_argument("--iou", type=probability, default=0.5)
    sam.add_argument("--max-det", type=positive, default=500)
    sam.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    sam.add_argument("--limit", type=positive)
    sam.set_defaults(overlap=0.25, imgsz=1008, tile_size=None, sam2_checkpoint=None)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "fetch-models":
            from .assets import ASSETS, download_asset

            for name in ASSETS if args.model == "all" else [args.model]:
                print(download_asset(name, args.output))
        elif args.command == "fetch-sam3":
            from .sam3_backend import fetch_checkpoint

            print(fetch_checkpoint(args.output, args.revision))
        elif args.command == "sam3-doctor":
            from .sam3_backend import doctor

            print(json.dumps(doctor(), indent=2))
        elif args.command == "fetch-encoders":
            from .learned_classifier import build_classifier

            names = (
                ["resnet18", "siglip"]
                if args.architecture == "all"
                else [args.architecture]
            )
            for name in names:
                model = build_classifier(name, revision=args.revision)
                print(json.dumps(model.metadata, indent=2))
                del model
        elif args.command == "prepare-crops":
            from .crop_dataset import prepare_crops

            data = prepare_crops(
                args.source,
                args.output,
                args.class_names,
                args.padding,
                args.groups,
                args.allow_split_overlap,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "class_counts": data["class_counts"],
                        "suspected_filename_families": len(
                            data["suspected_cross_split_filename_families"]
                        ),
                    },
                    indent=2,
                )
            )
        elif args.command in {"train-classifier", "extract-features"}:
            import torch
            from .classifier_training import extract_features, train_classifier

            torch.set_num_threads(args.threads)
            if args.command == "extract-features":
                data = extract_features(
                    args.dataset,
                    args.output,
                    args.batch_size,
                    args.workers,
                    args.device,
                    args.model_id,
                    args.revision,
                    not args.random_init,
                    args.gpus,
                )
                print(
                    json.dumps(
                        {
                            "status": "completed",
                            "output": str(args.output),
                            "metadata": data["metadata"],
                        },
                        indent=2,
                    )
                )
            else:
                data = train_classifier(
                    args.output,
                    args.architecture,
                    args.dataset,
                    args.features,
                    args.epochs,
                    args.batch_size,
                    args.workers,
                    args.learning_rate,
                    args.seed,
                    args.device,
                    args.resume,
                    args.checkpoint_every,
                    args.time_budget_seconds,
                    args.max_batches_per_run,
                    not args.random_init,
                    args.gpus,
                )
                print(json.dumps(data, indent=2))
                if data["status"] == "interrupted":
                    return 75
        elif args.command == "gallery":
            from .gallery import build_gallery

            print(build_gallery(args.run))
        elif args.command == "benchmark":
            from .benchmark import compare_runs
            from .visualization import write_report

            paths = {name: Path(path) for name, path in args.run}
            thresholds = {name: float(value) for name, value in args.threshold}
            if len(paths) != len(args.run) or len(thresholds) != len(args.threshold):
                raise ValueError("Duplicate model names in benchmark arguments")
            report = compare_runs(
                paths,
                thresholds,
                args.annotations,
                args.gt_coordinates,
                args.match_iou,
                args.max_det,
            )
            write_report(paths, report, args.output)
            print(
                f"Comparison: {args.output / 'index.html'}; quality status: {report['quality_status']}"
            )
        elif args.command in {"inspect-polycam", "prepare-polycam"}:
            from .polycam import inspect_archive, prepare_archive

            info = (
                inspect_archive(args.archive)
                if args.command == "inspect-polycam"
                else prepare_archive(args.archive, args.output)
            )
            print(json.dumps(info, indent=2))
        elif args.command in {"inspect-wall-bundle", "prepare-wall-bundle"}:
            from .wall_bundle import inspect_wall_bundle, prepare_wall_bundle

            info = (
                inspect_wall_bundle(args.directory)
                if args.command == "inspect-wall-bundle"
                else prepare_wall_bundle(args.directory, args.output)
            )
            print(json.dumps(info, indent=2))
        elif args.command == "evaluate-crops":
            import torch
            from .assets import sha256
            from .evaluation import evaluate_crops
            from .models import CropClassifier

            torch.set_num_threads(args.threads)
            if args.classifier == "trained":
                from .learned_classifier import LearnedCropClassifier

                classifier = LearnedCropClassifier(args.weights, args.device)
            else:
                classifier = CropClassifier(args.weights, args.device)
            report = evaluate_crops(
                args.dataset,
                args.output,
                classifier,
                {"checkpoint_sha256": sha256(args.weights)},
            )
            print(json.dumps(report, indent=2))
        else:
            from .assets import sha256
            from .models import CropClassifier, HoldDetector, Sam2Segmenter
            from .pipeline import collect_images, run_inference, validate_image_source

            if args.overlap >= 1:
                raise ValueError("Tile overlap must be less than 1")
            if args.output.exists():
                raise FileExistsError(
                    f"Output already exists: {args.output}; choose a new run directory"
                )
            validate_image_source(args.source)
            collect_images(args.source)
            if args.command == "sam3":
                from .sam3_backend import Sam3Detector

                hold_prompts = args.prompt or [
                    "climbing hold", "climbing grip", "climbing foothold",
                ]
                volume_prompts = args.volume_prompt or ["climbing volume"]
                prompts = ([('hold', prompt) for prompt in hold_prompts]
                           + [('volume', prompt) for prompt in volume_prompts])
                detector = Sam3Detector(args.weights, prompts, args.device, args.threads)
                segmenter = detector
            else:
                detector = HoldDetector(args.weights, args.device, args.threads)
                segmenter = (
                    Sam2Segmenter(args.sam2_checkpoint, args.sam2_config, args.device)
                    if args.sam2_checkpoint
                    else None
                )
            if args.classifier == "trained":
                from .learned_classifier import LearnedCropClassifier

                classifier = LearnedCropClassifier(args.classifier_weights, args.device)
            else:
                classifier = (
                    CropClassifier(args.classifier_weights, args.device)
                    if args.classifier == "cnn"
                    else None
                )
            metadata = {
                "detector": {"path": str(args.weights), "sha256": sha256(args.weights)},
                "classifier": args.classifier,
                "device": detector.device,
            }
            if args.command == "sam3":
                from .sam3_backend import SAM3_SOURCE_REVISION

                metadata["detector"].update(
                    {
                        "method": "sam3_text_prompt",
                        "prompts": prompts,
                        "source_revision": SAM3_SOURCE_REVISION,
                    }
                )
            if classifier:
                metadata["classifier_checkpoint"] = {
                    "path": str(args.classifier_weights),
                    "sha256": sha256(args.classifier_weights),
                }
                if args.classifier == "trained":
                    metadata["classifier_checkpoint"]["training"] = classifier.metadata
            if segmenter and args.command != "sam3":
                metadata["segmenter"] = {
                    "path": str(args.sam2_checkpoint),
                    "sha256": sha256(args.sam2_checkpoint),
                    "config": args.sam2_config,
                }
            summary = run_inference(
                args.source,
                args.output,
                detector,
                classifier,
                segmenter,
                confidence=args.conf,
                minimum_type_score=args.min_type_score,
                iou=args.iou,
                imgsz=args.imgsz,
                tile_size=args.tile_size or 0,
                overlap=args.overlap,
                max_det=args.max_det,
                rotate=args.rotate,
                limit=args.limit,
                model_metadata=metadata,
                infer_grip_types=args.classifier != "none",
            )
            print(f"Saved {summary['frame_count']} frames to {args.output}")
            print(f"View results: {args.output / 'index.html'}")
    except (OSError, ValueError, RuntimeError, ImportError, URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
