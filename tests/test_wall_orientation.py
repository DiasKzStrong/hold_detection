import json

import numpy as np
from PIL import Image
import trimesh

from hold_detection.cli import main
from hold_detection.wall_orientation import estimate_up


def test_upright_camera_poses_recover_vertical_despite_pitch():
    up = np.array([0.0, -0.6, 0.8])
    horizontal = np.array([0.0, 0.8, 0.6])
    cameras = {}
    for index, yaw in enumerate(np.linspace(-0.6, 0.7, 6)):
        forward = (np.cos(yaw) * horizontal + np.sin(yaw) * np.array([1, 0, 0])) * 0.8 + 0.6 * up
        right = np.cross(forward, up) / 0.8
        down = np.cross(forward, right)
        cameras[str(index)] = {"rotation": [right, down, forward]}
    result = estimate_up(cameras)
    assert result["status"] == "estimated"
    np.testing.assert_allclose(result["up_vector"], up, atol=1e-8)

    parallel = {str(index): cameras["0"] for index in range(6)}
    assert estimate_up(parallel)["reason"] == "camera_angles_too_similar"
    for camera in list(cameras.values())[3:]:
        camera["rotation"] = [-camera["rotation"][0], -camera["rotation"][1], camera["rotation"][2]]
    rejected = estimate_up(cameras)
    assert rejected["status"] == "manual_required"
    assert rejected["up_vector"] is None


def test_manual_vertical_reexports_existing_holds_and_rejects_invalid_depth(tmp_path, capsys):
    frame_path = tmp_path / "inference/frame_00000"
    frame_path.mkdir(parents=True)
    (frame_path.parent / "summary.json").write_text('{"frame_count": 1}')
    model = tmp_path / "camera_model"
    model.mkdir()
    (model / "cameras.txt").write_text("1 PINHOLE 32 24 16 16 16 12\n")
    (model / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame.jpg\n\n")
    depth = tmp_path / "frame.jpg.photometric.bin"
    values = np.full((24, 32), 2.0, dtype="<f4")
    values[2, 2] = 0
    depth.write_bytes(b"32&24&1&" + values.tobytes())
    mesh = tmp_path / "wall.ply"
    source_mesh = trimesh.creation.box(extents=[1, 3, 0.2]).apply_translation([0, -0.75, 2])
    source_mesh.export(mesh)
    raw_glb = tmp_path / "wall.glb"
    source_mesh.copy().apply_scale(0.5).export(raw_glb)
    raw_bytes = raw_glb.read_bytes()
    reconstruction = tmp_path / "reconstruction.json"
    reconstruction.write_text(json.dumps({
        "camera_model": str(model), "depth_directory": str(tmp_path),
        "mesh_path": str(mesh), "glb_path": str(raw_glb), "scale_m_per_unit": 0.5, "units": "meters",
    }))
    holds = []
    for index, top in enumerate([14, 5], 1):
        pixels = np.zeros((24, 32), dtype=np.uint8)
        pixels[top:top + 3, 12:15] = 255
        mask = f"hold{index}.png"
        Image.fromarray(pixels).save(frame_path / mask)
        holds.append({"hold_id": f"h{index}", "mask_path": mask, "detector_score": 0.9,
                      "bbox_xyxy": [12, top, 15, top + 3]})
    (frame_path / "holds.json").write_text(json.dumps({
        "frame_id": "frame_00000", "source_image": str(tmp_path / "frame.jpg"),
        "processed_size_wh": [32, 24], "holds": holds,
    }))

    args = ["export-wall", str(tmp_path), "--mode", "3d", "--reconstruction-path", str(reconstruction)]
    scene_path = tmp_path / "scene.json"
    assert main(args) == 0
    original = json.loads(scene_path.read_text())
    assert original["wall"]["up_orientation"]["status"] == "manual_required"
    assert all(hold["height_3d"] is None for hold in original["holds"])

    reference = tmp_path / "up_reference.json"
    clicks = {"image": "frame.jpg", "bottom_px": [16, 18], "top_px": [16, 4]}
    reference.write_text(json.dumps(clicks))
    assert main(args + ["--up-reference", str(reference)]) == 0
    calibrated = scene_path.read_text()
    scene = json.loads(calibrated)
    assert scene["wall"]["up_vector"] == [0, -1, 0]
    assert scene["wall"]["up_orientation"]["depth_map_type"] == "photometric"
    assert all(obs["geometry"]["depth_map_type"] == "photometric" for obs in scene["observations"])
    assert [h["id"] for h in scene["holds"]] == [h["id"] for h in original["holds"]]
    low, high = scene["holds"]
    assert high["height_3d"] > low["height_3d"]
    assert high["height_m"] == 0.5 * high["height_3d"]
    assert scene["wall"]["glb_up_axis"] == "y"
    upright = tmp_path / "wall_upright.glb"
    assert scene["wall"]["glb_path"] == str(upright)
    transform = np.asarray(scene["wall"]["reconstruction_to_glb"])
    aligned_mesh = trimesh.load_mesh(upright, process=False)
    np.testing.assert_allclose(aligned_mesh.vertices, trimesh.transform_points(source_mesh.vertices, transform))
    assert high["position_glb"][1] > low["position_glb"][1]
    np.testing.assert_allclose(high["position_glb"][1], high["height_m"])
    match = high["mesh_correspondence"]
    np.testing.assert_allclose(match["position_glb"], aligned_mesh.vertices[match["vertex_index"]])
    assert raw_glb.read_bytes() == raw_bytes
    assert (tmp_path / "wall_scene.json").read_text() == calibrated

    reference.write_text(json.dumps(dict(clicks, top_px=[2, 2])))
    assert main(args + ["--up-reference", str(reference)]) == 1
    assert "no valid stereo depth" in capsys.readouterr().err
    assert scene_path.read_text() == calibrated

    with (model / "images.txt").open("a") as poses:
        for index, yaw in enumerate([-0.6, 0.6], 2):
            poses.write(f"{index} {np.cos(yaw / 2)} 0 {np.sin(yaw / 2)} 0 0 0 0 1 view{index}.jpg\n\n")
    assert main(args) == 0
    auto = json.loads(scene_path.read_text())
    assert auto["wall"]["up_orientation"]["status"] == "estimated"
    assert auto["wall"]["glb_path"] == str(upright)
    np.testing.assert_allclose(auto["wall"]["reconstruction_to_glb"], transform, atol=1e-7)
    assert raw_glb.read_bytes() == raw_bytes
