"""Orchestrate wall analysis stages for a complete 2D or 3D run."""

from pathlib import Path
import signal

from .perception.segmentation import PipelinePaused, segment_capture


def analyze_wall(source, output, mode="2d", sam_weights=None, type_weights=None,
                 device="0", max_frames=48, colmap="colmap", scale_m_per_unit=None,
                 quality="medium", annotations=None, confidence=0.5, scale_reference=None,
                 type_imgsz=1024, type_min_score=0.5):
    """Run the complete photo/video workflow; resume completed preparation and reconstruction."""
    from .capture import prepare_capture
    from .reconstruction import reconstruct_wall
    from .scene import export_wall_scene
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop = output / "STOP_REQUESTED"
    if stop.exists():
        stop.unlink()

    def request_pause(signum, frame):
        stop.touch()

    previous = {sig: signal.signal(sig, request_pause) for sig in {signal.SIGTERM, signal.SIGUSR1}}
    try:
        capture = output / "capture"
        prepare_capture(source, capture, mode=mode, max_frames=max_frames)
        if stop.exists():
            raise PipelinePaused("Paused after input preparation")
        reconstruction_path = None
        if mode == "3d":
            reconstruct_wall(capture, output / "reconstruction", colmap, quality, scale_m_per_unit, scale_reference)
            reconstruction_path = output / "reconstruction/reconstruction.json"
        segment_capture(capture, output, reconstruction_path, sam_weights, type_weights, device,
                        confidence, stop, type_imgsz=type_imgsz, type_min_score=type_min_score)
        return export_wall_scene(output, mode, reconstruction_path, annotations)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
