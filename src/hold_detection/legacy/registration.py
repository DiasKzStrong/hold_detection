"""Quality-gated rigid registration from Polycam capture space to a GLB mesh."""

from __future__ import annotations

import json
from pathlib import Path
from itertools import permutations, product

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import trimesh


def load_glb_vertices(path: Path) -> np.ndarray:
    """Load GLB vertices with scene graph transforms baked by trimesh."""
    scene = trimesh.load_scene(path, process=False)
    mesh = scene.to_mesh()
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not len(vertices):
        raise ValueError("GLB contains no mesh POSITION data")
    return vertices


def capture_point_cloud(capture: Path, maximum_points: int = 8000) -> np.ndarray:
    """Build a deterministic high-confidence cloud from Polycam RGB-D keyframes."""
    keyframes = capture / "keyframes"
    camera_group = (
        keyframes / "corrected_cameras"
        if (keyframes / "corrected_cameras").is_dir() else keyframes / "cameras"
    )
    pieces = []
    for camera_path in sorted(camera_group.glob("*.json")):
        depth_path = keyframes / "depth" / f"{camera_path.stem}.png"
        confidence_path = keyframes / "confidence" / f"{camera_path.stem}.png"
        if not depth_path.is_file() or not confidence_path.is_file():
            continue
        camera = json.loads(camera_path.read_text())
        depth = np.asarray(Image.open(depth_path), dtype=float) / 1000.0
        confidence = np.asarray(Image.open(confidence_path))
        rows, columns = np.nonzero((depth > 0) & (confidence == confidence.max()))
        if not len(rows):
            continue
        step = max(1, len(rows) // 500)
        rows, columns = rows[::step], columns[::step]
        z = depth[rows, columns]
        scale_x = camera["width"] / depth.shape[1]
        scale_y = camera["height"] / depth.shape[0]
        points = np.column_stack([
            ((columns + 0.5) * scale_x - camera["cx"]) * z / camera["fx"],
            ((rows + 0.5) * scale_y - camera["cy"]) * z / camera["fy"],
            z,
        ])
        rotation = np.array([
            [camera[f"t_{row}{column}"] for column in range(3)]
            for row in range(3)
        ])
        translation = np.array([camera[f"t_{row}3"] for row in range(3)])
        pieces.append(points @ rotation.T + translation)
    if not pieces:
        raise ValueError("Capture has no complete camera/depth/confidence frames")
    cloud = np.concatenate(pieces)
    if len(cloud) > maximum_points:
        indices = np.linspace(0, len(cloud) - 1, maximum_points, dtype=int)
        cloud = cloud[indices]
    return cloud


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _icp(
    source: np.ndarray, target: np.ndarray, rotation: np.ndarray,
    translation: np.ndarray, trim_fraction: float = 0.35, iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray, float]:
    tree = cKDTree(target)
    previous = np.inf
    for _ in range(iterations):
        transformed = source @ rotation.T + translation
        distances, indices = tree.query(transformed)
        threshold = np.quantile(distances, trim_fraction)
        keep = distances <= threshold
        update_rotation, update_translation = _rigid_fit(
            transformed[keep], target[indices[keep]]
        )
        rotation = update_rotation @ rotation
        translation = translation @ update_rotation.T + update_translation
        error = float(np.sqrt(np.mean(np.square(distances[keep]))))
        if abs(previous - error) < 1e-6:
            break
        previous = error
    return rotation, translation, previous


def register_capture_to_glb(capture: Path, glb: Path) -> dict:
    """Estimate capture-to-GLB rigid transform and report quality without hiding failure."""
    source = capture_point_cloud(capture)
    target = load_glb_vertices(glb)
    if len(target) > 12000:
        target = target[np.linspace(0, len(target) - 1, 12000, dtype=int)]
    source_center, target_center = np.mean(source, axis=0), np.mean(target, axis=0)
    _, source_basis = np.linalg.eigh(np.cov(source - source_center, rowvar=False))
    _, target_basis = np.linalg.eigh(np.cov(target - target_center, rowvar=False))
    source_basis, target_basis = source_basis[:, ::-1], target_basis[:, ::-1]
    tree = cKDTree(target)
    candidates = []
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            basis = target_basis[:, permutation] @ np.diag(signs)
            rotation = basis @ source_basis.T
            if np.linalg.det(rotation) < 0:
                continue
            translation = target_center - source_center @ rotation.T
            distances, _ = tree.query(source @ rotation.T + translation)
            score = float(np.mean(np.partition(distances, int(0.3 * len(distances)))[:int(0.3 * len(distances))]))
            candidates.append((score, rotation, translation))
    refined = [
        _icp(source, target, rotation, translation)
        for _, rotation, translation in sorted(candidates, key=lambda item: item[0])[:6]
    ]
    rotation, translation, trimmed_rmse = min(refined, key=lambda item: item[2])
    transformed = source @ rotation.T + translation
    distances, _ = tree.query(transformed)
    inlier_5cm = float(np.mean(distances <= 0.05))
    inlier_10cm = float(np.mean(distances <= 0.10))
    accepted = inlier_10cm >= 0.20 and trimmed_rmse <= 0.08
    quality = (
        "high" if inlier_10cm >= 0.50 and trimmed_rmse <= 0.04
        else "medium" if inlier_10cm >= 0.35 and trimmed_rmse <= 0.06
        else "low"
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return {
        "status": "registered" if accepted else "rejected",
        "method": "pca_initialized_trimmed_point_to_point_icp",
        "quality": quality if accepted else "rejected",
        "capture_to_glb_matrix_row_major": matrix.round(9).tolist(),
        "trimmed_rmse_m": round(float(trimmed_rmse), 6),
        "inlier_fraction_5cm": round(inlier_5cm, 6),
        "inlier_fraction_10cm": round(inlier_10cm, 6),
        "source_point_count": len(source),
        "target_vertex_count": len(target),
        "acceptance": "inlier_fraction_10cm >= 0.20 and trimmed_rmse_m <= 0.08",
    }


def transform_points(points: np.ndarray, registration: dict) -> np.ndarray:
    matrix = np.asarray(registration["capture_to_glb_matrix_row_major"], dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def match_points_to_glb(
    points_capture: np.ndarray, glb: Path, registration: dict,
    vertices: np.ndarray | None = None,
) -> dict:
    """Transform hold samples and summarize their nearest GLB vertex anchors."""
    points_glb = transform_points(points_capture, registration)
    vertices = load_glb_vertices(glb) if vertices is None else vertices
    distances, indices = cKDTree(vertices).query(points_glb)
    anchors = vertices[indices]
    return {
        "method": "registered_points_to_nearest_glb_vertices",
        "sample_count": len(points_glb),
        "anchor_position_glb_m": np.median(anchors, axis=0).round(6).tolist(),
        "point_to_mesh_distance_median_m": round(float(np.median(distances)), 6),
        "point_to_mesh_distance_p90_m": round(float(np.quantile(distances, 0.9)), 6),
        "needs_review": registration.get("quality") != "high",
    }
