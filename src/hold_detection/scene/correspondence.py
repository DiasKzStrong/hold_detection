"""Reviewable one-to-one correspondence to named holds in an annotated GLB."""

import json
from pathlib import Path
import re

import numpy as np
from scipy.optimize import linear_sum_assignment
import trimesh


def match_annotated_glb(scene_path, annotated_glb, transform, output, maximum_distance_m=0.1):
    """Use an explicit scene-to-annotation transform; never assume independently reconstructed frames align."""
    scene_path, output = Path(scene_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new correspondence output path")
    scene = json.loads(scene_path.read_text())
    matrix = np.asarray(json.loads(Path(transform).read_text()), dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12):
        raise ValueError("Expected a finite invertible 4x4 scene-to-annotation transform")
    if not 0 < maximum_distance_m < float("inf"):
        raise ValueError("Maximum correspondence distance must be positive and finite")
    annotated = trimesh.load_scene(annotated_glb, process=False)
    targets = []
    for node in sorted(annotated.graph.nodes_geometry):
        pose, geometry = annotated.graph[node]
        mesh = annotated.geometry[geometry]
        identifier = mesh.metadata.get("hold_id") or str(node)
        if not re.search(r"hold|volume", identifier, re.IGNORECASE) or not len(mesh.vertices):
            continue
        points = trimesh.transform_points(mesh.vertices, pose)
        targets.append({"id": identifier, "position_m": np.median(points, axis=0).tolist()})
    if not targets:
        raise ValueError("Annotated GLB has no named hold/volume objects; preserve object IDs during export")
    sources = [hold for hold in scene["holds"] if hold.get("position_3d") is not None]
    for hold in scene["holds"]:
        hold["ground_truth_match"] = None
    if sources:
        points = trimesh.transform_points([h["position_3d"] for h in sources], matrix)
        centers = np.array([t["position_m"] for t in targets])
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        for row, column in zip(*linear_sum_assignment(np.where(distances <= maximum_distance_m, distances, 1e6))):
            distance = distances[row, column]
            if distance <= maximum_distance_m:
                sources[row]["ground_truth_match"] = {
                    "annotated_hold_id": targets[column]["id"], "distance_m": float(distance),
                    "method": "aligned_centroid_hungarian", "needs_review": True,
                }
    matched = sum(hold["ground_truth_match"] is not None for hold in sources)
    scene["annotated_glb_correspondence"] = {
        "annotated_glb": str(Path(annotated_glb).resolve()), "matched_holds": matched,
        "annotated_holds": len(targets), "unmatched_scene_holds": len(scene["holds"]) - matched,
        "scene_to_annotation_matrix": matrix.tolist(), "maximum_distance_m": maximum_distance_m,
        "coordinate_contract": "input_scene_reconstruction_units_to_annotation_meters",
        "status": "needs_review",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    return scene["annotated_glb_correspondence"]
