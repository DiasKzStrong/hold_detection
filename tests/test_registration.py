from pathlib import Path

import numpy as np
import pytest
import trimesh

from hold_detection.registration import (
    load_glb_vertices,
    match_points_to_glb,
    transform_points,
)


def test_trimesh_glb_scene_transforms_and_mesh_matching(tmp_path: Path):
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    transform = np.eye(4)
    transform[:3, 3] = (3.0, -2.0, 1.0)
    scene.add_geometry(box, transform=transform)
    glb = tmp_path / "wall.glb"
    glb.write_bytes(scene.export(file_type="glb"))

    vertices = load_glb_vertices(glb)
    assert np.ptp(vertices, axis=0) == pytest.approx([2.0, 4.0, 6.0])
    assert np.mean(vertices, axis=0) == pytest.approx([3.0, -2.0, 1.0])

    registration = {
        "capture_to_glb_matrix_row_major": np.eye(4).tolist(),
        "quality": "high",
    }
    samples = vertices[:8]
    assert transform_points(samples, registration) == pytest.approx(samples)
    match = match_points_to_glb(samples, glb, registration, vertices)
    assert match["point_to_mesh_distance_median_m"] == 0
    assert match["point_to_mesh_distance_p90_m"] == 0
    assert not match["needs_review"]
