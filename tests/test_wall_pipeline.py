"""Public wall-analysis stages with real images, depth and mesh files."""

import json
import importlib.util
import ast
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import trimesh
import nbformat

from hold_detection.capture import prepare_capture
from hold_detection.geometry import Detection
from hold_detection.pipeline import run_inference
from hold_detection.reconstruction import lift_mask, read_colmap_depth, read_colmap_cameras, reconstruct_wall, scale_from_reference
from hold_detection.wall_scene import assign_type_proposals, build_wall_scene, export_wall_scene

ROOT = Path(__file__).resolve().parents[1]


def test_wall_notebook_is_a_valid_thin_library_client(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("wall_notebook", ROOT / "scripts/build_wall_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = ROOT / "notebooks/DeMoSearch_Wall_Analysis.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert json.loads(path.read_text()) == builder.build_wall_notebook()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None and cell.outputs == []
    analysis = next(cell.source for cell in notebook.cells if cell.id == "analyze")
    assert "analyze-wall" in analysis and "run_holds" in analysis


def test_every_slurm_entrypoint_has_valid_shell_syntax():
    for path in (ROOT / "scripts/slurm").iterdir():
        if path.suffix in {".sbatch", ".sh"}:
            subprocess.run(["bash", "-n", str(path)], check=True)


def observation(identity="one", frame="a", geometry=None):
    return {"id": identity, "frame_id": frame, "object_kind": "hold",
            "position_2d_normalized": [0.5, 0.5], "size_2d_normalized": [0.1, 0.1],
            "color_rgb": [10, 20, 180], "color_group_id": "color_1",
            "hold_type": "jug", "hold_type_confidence": 0.9,
            "hold_type_source": "yolo26_rgb_joint_score", "geometry": geometry}


def camera():
    return {"width": 8, "height": 8, "fx": 4, "fy": 4, "cx": 4, "cy": 4,
            "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}


def test_photo_capture_is_oriented_bounded_and_reusable(tmp_path):
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (300, 100), "red").save(photo)
    result = prepare_capture(photo, tmp_path / "capture", max_image_edge=100)
    assert result["settings"]["mode"] == "2d"
    assert Image.open(result["image_source"] + "/frame_00000.jpg").size == (100, 33)
    assert prepare_capture(photo, tmp_path / "capture", max_image_edge=100) == result
    with pytest.raises(ValueError, match="changed"):
        prepare_capture(photo, tmp_path / "capture", max_image_edge=120)


def test_video_capture_extracts_multiple_views_without_a_glb(tmp_path):
    import cv2
    video = tmp_path / "wall.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 24))
    for index in range(20):
        writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    result = prepare_capture(video, tmp_path / "capture", mode="3d", max_frames=5)
    assert result["input_kind"] == "video"
    assert len(result["frames"]) == 5
    assert result["frames"][-1]["timestamp_seconds"] == 3.8


def test_colmap_depth_format_and_world_projection_have_known_coordinates(tmp_path):
    path = tmp_path / "depth.bin"
    path.write_bytes(b"3&2&1&" + np.array([1, 2, 3, 4, 5, 6], dtype="<f4").tobytes())
    np.testing.assert_array_equal(read_colmap_depth(path), [[1, 2, 3], [4, 5, 6]])
    mask = np.zeros((8, 8), bool)
    mask[3:5, 3:5] = True
    pose = camera()
    pose["translation"] = [1, 0, 0]
    result = lift_mask(np.full((8, 8), 2.0), mask, pose)
    np.testing.assert_allclose(result["position"], [-1, 0, 2], atol=1e-8)
    assert result["protrusion"] == 0
    assert result["wall_normal"][2] == -1


def test_colmap_pose_reader_preserves_blank_track_lines_and_names(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 PINHOLE 8 8 4 4 4 4\n")
    (tmp_path / "images.txt").write_text("1 1 0 0 0 1 2 3 1 first.jpg\n\n2 1 0 0 0 0 0 0 1 second.jpg\n\n")
    cameras = read_colmap_cameras(tmp_path)
    assert set(cameras) == {"first.jpg", "second.jpg"}
    assert cameras["first.jpg"]["translation"] == [1, 2, 3]


def test_sam_instances_receive_at_most_one_valid_type_proposal():
    holds = [{"bbox_xyxy": [0, 0, 10, 10]}, {"bbox_xyxy": [8, 0, 18, 10]}]
    result = assign_type_proposals(holds, [{"bbox_xyxy": [0, 0, 10, 10], "score": 0.9, "label": "jug"}])
    assert [hold["hold_type"] for hold in result] == ["jug", "unknown"]
    assert len(result) == 2


def test_two_modes_share_schema_and_uncalibrated_3d_never_claims_meters():
    geometry = {"samples": [[0, 0, 2], [0.1, 0, 2], [0, 0.1, 2]],
                "extent": [0.1, 0.1, 0], "wall_normal": [0, 0, -1], "protrusion": 0.02}
    first = observation(geometry=geometry)
    second = observation("two", "b", geometry)
    unmapped = observation("three", "c")
    two_d = build_wall_scene([first], "2d")
    three_d = build_wall_scene([first, second, unmapped], "3d", {"units": "arbitrary"})
    assert set(two_d["holds"][0]) == set(three_d["holds"][0])
    assert len(three_d["holds"]) == 2  # Same Physical Hold in two views, plus retained unmapped hold.
    assert three_d["holds"][0]["observation_count"] == 2
    assert three_d["holds"][0]["position_3d_m"] is None
    assert three_d["holds"][0]["volume_m3"] is None
    assert three_d["review"]["unmapped_hold_count"] == 1
    # Even if an RGB-D fixture is supplied, 2D export must not leak 3D measurements.
    assert two_d["holds"][0]["position_3d"] is None


def test_single_view_holds_and_human_type_corrections_are_not_discarded():
    scene = build_wall_scene([observation()], "2d")
    identity = scene["holds"][0]["id"]
    corrected = build_wall_scene([observation()], "2d", annotations={identity: {"hold_type": "undercling", "is_start": True}})
    assert corrected["holds"][0]["hold_type"] == "undercling"
    assert corrected["holds"][0]["hold_type_source"] == "human"
    assert corrected["holds"][0]["is_start"] is True


def test_known_reference_recovers_scale_without_inventing_missing_depth():
    depth = np.full((8, 8), 2.0)
    reference = {"point_a_px": [2, 4], "point_b_px": [6, 4], "distance_m": 1.0}
    assert scale_from_reference(depth, camera(), reference) == 0.5
    depth[4, 2] = 0
    with pytest.raises(ValueError, match="valid stereo depth"):
        scale_from_reference(depth, camera(), reference)


def test_reconstruction_external_adapter_exports_real_mesh_and_reuses_completed_stage(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for index in range(3):
        Image.new("RGB", (8, 8), (index * 30, 20, 90)).save(photos / f"{index}.jpg")
    capture = tmp_path / "capture"
    prepare_capture(photos, capture, mode="3d")
    external = tmp_path / "colmap"
    external.write_text(f"#!{sys.executable}\n" + '''
import sys, shutil
from pathlib import Path
import trimesh
args = sys.argv[1:]
def option(name): return Path(args[args.index(name) + 1])
if args[0] == 'automatic_reconstructor':
    dense = option('--workspace_path') / 'dense/0'
    (dense / 'sparse').mkdir(parents=True, exist_ok=True)
    shutil.copytree(option('--image_path'), dense / 'images', dirs_exist_ok=True)
    trimesh.creation.box().export(dense / 'meshed-poisson.ply')
    (dense / 'fused.ply').write_text('fixture cloud')
    (dense / 'stereo/depth_maps').mkdir(parents=True, exist_ok=True)
elif args[0] == 'model_converter':
    target = option('--output_path')
    (target / 'cameras.txt').write_text('1 PINHOLE 8 8 4 4 4 4\\n')
    (target / 'images.txt').write_text(''.join(f'{i+1} 1 0 0 0 0 0 0 1 frame_{i:05d}.jpg\\n\\n' for i in range(3)))
else:
    raise SystemExit(2)
''')
    external.chmod(0o755)
    result = reconstruct_wall(capture, tmp_path / "reconstruction", str(external), scale_m_per_unit=2.0)
    loaded = trimesh.load_scene(result["glb_path"]).to_mesh()
    np.testing.assert_allclose(loaded.extents, [2, 2, 2])
    assert result["registered_images"] == 3
    external.unlink()  # Completed-stage reuse needs no external process.
    assert reconstruct_wall(capture, tmp_path / "reconstruction", str(external), scale_m_per_unit=2.0) == result


def test_annotated_glb_matching_requires_alignment_and_keeps_stable_target_ids(tmp_path):
    from hold_detection.correspondence import match_annotated_glb
    mesh = trimesh.Scene()
    mesh.add_geometry(trimesh.creation.box(), node_name="blue_v5_holds.004", geom_name="hold", transform=np.eye(4))
    mesh.export(tmp_path / "marked.glb")
    matrix = tmp_path / "transform.json"
    matrix.write_text(json.dumps(np.eye(4).tolist()))
    source = tmp_path / "scene.json"
    source.write_text(json.dumps({"holds": [{"id": "ours", "position_3d": [0, 0, 0]},
                                           {"id": "far", "position_3d": [10, 0, 0]}]}))
    result = match_annotated_glb(source, tmp_path / "marked.glb", matrix, tmp_path / "matched.json")
    assert result["matched_holds"] == 1
    matched = json.loads((tmp_path / "matched.json").read_text())
    assert matched["holds"][0]["ground_truth_match"]["annotated_hold_id"] == "blue_v5_holds.004"
    assert matched["holds"][1]["ground_truth_match"] is None


def test_segmentation_reuses_completed_frame_processes_and_exports_final_scene(tmp_path, monkeypatch):
    from hold_detection import models
    from hold_detection.wall_pipeline import segment_capture
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (20, 20), "blue").save(source)
    capture = tmp_path / "capture"
    prepare_capture(source, capture)
    weights = tmp_path / "sam3.pt"
    weights.write_bytes(b'external model fixture')
    type_weights = tmp_path / "types.pt"
    type_weights.write_bytes(b'external hold-type model fixture')

    class Detector:
        def detect(self, image, **settings):
            return [Detection((2, 2, 9, 9), 0.9, "hold", label_is_type=False)]

    class TypeDetector:
        def __init__(self, weights, device):
            assert weights == type_weights

        def detect(self, image, **settings):
            assert settings["imgsz"] == 640
            return [Detection((2, 2, 9, 9), 0.8, "jug")]

    def model_process(command, **kwargs):
        assert "--no-finalize" in command and command[3] == "sam3"
        run_inference(Path(command[4]), Path(command[command.index("--output") + 1]), Detector(),
                      start=int(command[command.index("--start") + 1]), limit=1, finalize=False)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", model_process)
    monkeypatch.setattr(models, "HoldDetector", TypeDetector)
    run = tmp_path / "run"
    segment_capture(capture, run, sam_weights=weights, type_weights=type_weights, type_imgsz=640)
    initial = export_wall_scene(run, "2d")
    assert initial["holds"][0]["hold_type"] == "jug"

    def unexpected_process(*args, **kwargs):
        raise AssertionError("A completed inference frame must not run again")

    monkeypatch.setattr(subprocess, "run", unexpected_process)
    segment_capture(capture, run, sam_weights=weights, type_weights=type_weights, type_imgsz=640)
    assert export_wall_scene(run, "2d") == initial
    assert (run / "inference/index.html").is_file()
    with pytest.raises(ValueError, match="settings changed"):
        segment_capture(capture, run, sam_weights=weights, type_weights=type_weights, type_imgsz=1024)


def test_batch_inference_can_resume_a_slice_without_finalizing(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ["a.jpg", "b.jpg"]:
        Image.new("RGB", (20, 20), "blue").save(images / name)

    class Detector:
        def detect(self, image, **settings):
            return [Detection((2, 2, 9, 9), 0.9, "hold", label_is_type=False)]

    output = tmp_path / "run/inference"
    summary = run_inference(images, output, Detector(), start=1, limit=1, finalize=False)
    assert summary["frame_count"] == 1
    assert Path(summary["frames"][0]["source_image"]).name == "b.jpg"
    assert not (output / "scene.json").exists()
    scene = export_wall_scene(tmp_path / "run", "2d")
    assert len(scene["holds"]) == 1
    assert (tmp_path / "run/scene.json").read_bytes() == (tmp_path / "run/wall_scene.json").read_bytes()


@pytest.mark.parametrize("mode,stages", [("2d", ["prepare", "segment", "export"]),
                                        ("3d", ["prepare", "reconstruct", "segment", "export"])])
def test_sbatch_submission_orders_stages_with_afterok_dependencies(tmp_path, mode, stages):
    binary = tmp_path / "sbatch"
    log = tmp_path / "jobs.jsonl"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\np=Path({str(log)!r})\n"
                      "rows=p.read_text().splitlines() if p.exists() else []\n"
                      "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nprint(100+len(rows))\n")
    binary.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/submit_wall.sh")], env={
        **os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "DEMO_PROJECT": str(ROOT), "DEMO_SOURCE": "/fake/wall.mov", "DEMO_RUN": "/fake/run",
        "DEMO_PYTHON": "/fake/python", "DEMO_SAM_WEIGHTS": "/fake/sam3.pt", "DEMO_MODE": mode,
    }, capture_output=True, text=True, check=True)
    jobs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(jobs) == len(stages)
    for index, stage in enumerate(stages):
        assert f"--export=ALL,DEMO_STAGE={stage}" in jobs[index]
        if index:
            assert f"--dependency=afterok:{99 + index}" in jobs[index]
    assert "JSON export" in result.stdout
