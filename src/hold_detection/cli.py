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
    fetch = commands.add_parser("fetch-models", help="Download verified reference checkpoints")
    fetch.add_argument("--output", type=Path, default=Path("models"))
    fetch.add_argument("--model", choices=("detector", "classifier", "all"), default="all")
    sam_fetch = commands.add_parser("fetch-sam3", help="Download gated official facebook/sam3 weights")
    sam_fetch.add_argument("--output", type=Path, default=Path("models/sam3"))
    sam_fetch.add_argument("--revision", default="main")
    commands.add_parser("sam3-doctor", help="Report SAM 3 software, GPU and auth prerequisites")
    gallery = commands.add_parser("gallery", help="Build an offline HTML gallery for a completed run")
    gallery.add_argument("run", type=Path)
    benchmark = commands.add_parser("benchmark", help="Compare shared-image runs; optional COCO quality metrics")
    benchmark.add_argument("--run", nargs=2, action="append", metavar=("MODEL", "PATH"), required=True)
    benchmark.add_argument("--threshold", nargs=2, action="append", metavar=("MODEL", "SCORE"), required=True)
    benchmark.add_argument("--annotations", type=Path)
    benchmark.add_argument("--gt-coordinates", choices=("source", "processed"), default="source")
    benchmark.add_argument("--match-iou", type=probability, default=0.5)
    benchmark.add_argument("--max-det", type=positive, default=500)
    benchmark.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect-polycam", help="Inspect a raw-data ZIP without extracting")
    inspect.add_argument("archive", type=Path)
    prepare = commands.add_parser("prepare-polycam", help="Unpack RGB-D frames and camera metadata")
    prepare.add_argument("archive", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-crops", help="Evaluate the CNN on labeled crops")
    evaluate.add_argument("dataset", type=Path, help="Directory with images/ and labels/<stem>.txt")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, default=Path("models/climbing-cnn.pth"))
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--threads", type=positive, default=4)
    predict = commands.add_parser("predict", help="Find holds, predict types, optionally refine masks")
    predict.add_argument("source", type=Path, help="An image or directory of RGB images")
    predict.add_argument("--output", type=Path, required=True, help="A NEW run directory")
    predict.add_argument("--weights", type=Path, default=Path("models/climbing-yolo11s.pt"))
    predict.add_argument("--classifier", choices=("detector", "cnn"), default="detector")
    predict.add_argument("--classifier-weights", type=Path, default=Path("models/climbing-cnn.pth"))
    predict.add_argument("--device", default="auto")
    predict.add_argument("--threads", type=positive, default=4)
    predict.add_argument("--conf", type=probability, default=0.35)
    predict.add_argument("--min-type-score", type=probability, default=0.5)
    predict.add_argument("--iou", type=probability, default=0.5)
    predict.add_argument("--imgsz", type=positive, default=1024)
    predict.add_argument("--tile-size", type=positive, help="Also detect on overlapping image tiles")
    predict.add_argument("--overlap", type=probability, default=0.25)
    predict.add_argument("--max-det", type=positive, default=500)
    predict.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0,
                         help="Clockwise rotation before detection; output uses rotated coordinates")
    predict.add_argument("--limit", type=positive)
    predict.add_argument("--sam2-checkpoint", type=Path)
    predict.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    sam = commands.add_parser("sam3", help="Find holds with text-prompted SAM 3 masks; optionally classify")
    sam.add_argument("source", type=Path)
    sam.add_argument("--output", type=Path, required=True)
    sam.add_argument("--weights", type=Path, default=Path("models/sam3/sam3.pt"))
    sam.add_argument("--prompt", default="climbing hold")
    sam.add_argument("--classifier", choices=("none", "cnn"), default="cnn")
    sam.add_argument("--classifier-weights", type=Path, default=Path("models/climbing-cnn.pth"))
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
            report = compare_runs(paths, thresholds, args.annotations, args.gt_coordinates,
                                  args.match_iou, args.max_det)
            write_report(paths, report, args.output)
            print(f"Comparison: {args.output / 'index.html'}; quality status: {report['quality_status']}")
        elif args.command in {"inspect-polycam", "prepare-polycam"}:
            from .polycam import inspect_archive, prepare_archive
            info = (inspect_archive(args.archive) if args.command == "inspect-polycam"
                    else prepare_archive(args.archive, args.output))
            print(json.dumps(info, indent=2))
        elif args.command == "evaluate-crops":
            import torch
            from .assets import sha256
            from .evaluation import evaluate_crops
            from .models import CropClassifier
            torch.set_num_threads(args.threads)
            classifier = CropClassifier(args.weights, args.device)
            report = evaluate_crops(args.dataset, args.output, classifier,
                                    {"checkpoint_sha256": sha256(args.weights)})
            print(json.dumps(report, indent=2))
        else:
            from .assets import sha256
            from .models import CropClassifier, HoldDetector, Sam2Segmenter
            from .pipeline import collect_images, run_inference, validate_image_source
            if args.overlap >= 1:
                raise ValueError("Tile overlap must be less than 1")
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}; choose a new run directory")
            validate_image_source(args.source)
            collect_images(args.source)
            if args.command == "sam3":
                from .sam3_backend import Sam3Detector
                detector = Sam3Detector(args.weights, args.prompt, args.device, args.threads)
                segmenter = detector
            else:
                detector = HoldDetector(args.weights, args.device, args.threads)
                segmenter = (Sam2Segmenter(args.sam2_checkpoint, args.sam2_config, args.device)
                             if args.sam2_checkpoint else None)
            classifier = (CropClassifier(args.classifier_weights, args.device)
                          if args.classifier == "cnn" else None)
            metadata = {"detector": {"path": str(args.weights), "sha256": sha256(args.weights)},
                        "classifier": args.classifier, "device": detector.device}
            if args.command == "sam3":
                from .sam3_backend import SAM3_SOURCE_REVISION
                metadata["detector"].update({
                    "method": "sam3_text_prompt", "prompt": args.prompt,
                    "source_revision": SAM3_SOURCE_REVISION,
                })
            if classifier:
                metadata["classifier_checkpoint"] = {
                    "path": str(args.classifier_weights), "sha256": sha256(args.classifier_weights),
                }
            if segmenter and args.command != "sam3":
                metadata["segmenter"] = {
                    "path": str(args.sam2_checkpoint), "sha256": sha256(args.sam2_checkpoint),
                    "config": args.sam2_config,
                }
            summary = run_inference(
                args.source, args.output, detector, classifier, segmenter,
                confidence=args.conf, minimum_type_score=args.min_type_score,
                iou=args.iou, imgsz=args.imgsz, tile_size=args.tile_size or 0,
                overlap=args.overlap, max_det=args.max_det, rotate=args.rotate,
                limit=args.limit, model_metadata=metadata,
            )
            print(f"Saved {summary['frame_count']} frames to {args.output}")
            print(f"View results: {args.output / 'index.html'}")
    except (OSError, ValueError, RuntimeError, ImportError, URLError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
