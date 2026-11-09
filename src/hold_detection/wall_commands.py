"""Argument parsing for training and complete wall-analysis stages."""

import json
from pathlib import Path


def add_wall_commands(commands):
    download = commands.add_parser("download-yolo", help="Download HoldClassificatore from Roboflow using an environment key")
    download.add_argument("--output", type=Path, required=True)
    download.add_argument("--version", type=int, required=True)
    download.add_argument("--project", choices=["holdclassificatore_v3_mix", "holdclassificatore_v2.2"], default="holdclassificatore_v3_mix")
    match = commands.add_parser("match-annotated-glb", help="Match reconstructed holds to annotated GLB object IDs")
    match.add_argument("scene_path", type=Path)
    match.add_argument("--annotated-glb", type=Path, required=True)
    match.add_argument("--transform", type=Path, required=True)
    match.add_argument("--output", type=Path, required=True)
    match.add_argument("--maximum-distance-m", type=float, default=0.1)
    prepare = commands.add_parser("prepare-yolo", help="Validate a HoldClassificatore YOLO detection export")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train-yolo", help="Fine-tune the six-class YOLO26 detector")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", default="yolo26m.pt")
    for name, default in [("epochs", 100), ("imgsz", 1024), ("batch", 8), ("workers", 2)]:
        train.add_argument(f"--{name}", type=int, default=default)
    train.add_argument("--device", default="0")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-file", type=Path)
    evaluate = commands.add_parser("evaluate-yolo", help="Evaluate validation-selected YOLO weights")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=["val", "test"], default="val")
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--imgsz", type=int, default=1024)

    capture = commands.add_parser("prepare-capture", help="Prepare a single photo, multiple photos or a video")
    capture.add_argument("source", type=Path)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--mode", choices=["2d", "3d"], required=True)
    capture.add_argument("--max-frames", type=int, default=48)
    capture.add_argument("--video-stride-seconds", type=float, default=0.75)
    capture.add_argument("--max-image-edge", type=int, default=1600)
    reconstruct = commands.add_parser("reconstruct-wall", help="Reconstruct an ordinary RGB capture with CUDA COLMAP")
    reconstruct.add_argument("capture", type=Path)
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--colmap", default="colmap")
    reconstruct.add_argument("--quality", choices=["low", "medium", "high", "extreme"], default="medium")
    reconstruct.add_argument("--scale-m-per-unit", type=float)
    reconstruct.add_argument("--scale-reference", type=Path)
    segment = commands.add_parser("segment-capture", help="Resume per-frame SAM3 segmentation and YOLO type proposals")
    segment.add_argument("capture", type=Path)
    segment.add_argument("--run", type=Path, required=True)
    segment.add_argument("--reconstruction-path", type=Path)
    segment.add_argument("--sam-weights", type=Path, required=True)
    segment.add_argument("--type-weights", type=Path)
    segment.add_argument("--type-imgsz", type=int, default=1024, help="YOLO hold-type inference input size")
    segment.add_argument("--device", default="0")
    segment.add_argument("--confidence", type=float, default=0.5)
    segment.add_argument("--stop-file", type=Path)
    export = commands.add_parser("export-wall", help="Fuse observations and export the common scene.json contract")
    export.add_argument("run", type=Path)
    export.add_argument("--mode", choices=["2d", "3d"], required=True)
    export.add_argument("--reconstruction-path", type=Path)
    export.add_argument("--annotations-path", type=Path)
    export.add_argument("--up-reference", dest="up_reference_path", type=Path,
                        help="JSON with the image name and bottom_px/top_px on a real vertical line")
    analyze = commands.add_parser("analyze-wall", help="Complete 2D photo or 3D photos/video wall analysis")
    analyze.add_argument("source", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--mode", choices=["2d", "3d"], required=True)
    analyze.add_argument("--sam-weights", type=Path, required=True)
    analyze.add_argument("--type-weights", type=Path)
    analyze.add_argument("--type-imgsz", type=int, default=1024, help="YOLO hold-type inference input size")
    analyze.add_argument("--device", default="0")
    analyze.add_argument("--max-frames", type=int, default=48)
    analyze.add_argument("--colmap", default="colmap")
    analyze.add_argument("--scale-m-per-unit", type=float)
    analyze.add_argument("--scale-reference", type=Path)
    analyze.add_argument("--quality", choices=["low", "medium", "high", "extreme"], default="medium")
    analyze.add_argument("--annotations", type=Path)
    analyze.add_argument("--confidence", type=float, default=0.5)


def dispatch_wall_command(args):
    implementations = {
        "download-yolo": ("yolo_training", "download_roboflow_dataset"),
        "match-annotated-glb": ("correspondence", "match_annotated_glb"),
        "prepare-yolo": ("yolo_training", "prepare_yolo_dataset"),
        "train-yolo": ("yolo_training", "train_yolo26"),
        "evaluate-yolo": ("yolo_training", "evaluate_yolo26"),
        "prepare-capture": ("capture", "prepare_capture"),
        "reconstruct-wall": ("reconstruction", "reconstruct_wall"),
        "segment-capture": ("wall_pipeline", "segment_capture"),
        "export-wall": ("wall_scene", "export_wall_scene"),
        "analyze-wall": ("wall_pipeline", "analyze_wall"),
    }
    if args.command not in implementations:
        return None
    from importlib import import_module
    from .wall_pipeline import PipelinePaused
    from .yolo_training import TrainingPaused
    module, name = implementations[args.command]
    parameters = vars(args).copy()
    parameters.pop("command")
    try:
        result = getattr(import_module(f"hold_detection.{module}"), name)(**parameters)
    except (TrainingPaused, PipelinePaused) as error:
        print(error, flush=True)
        return 75
    print(json.dumps(result, indent=2))
    return 0
