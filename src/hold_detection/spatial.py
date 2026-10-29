"""Metric spatial evidence from Polycam RGB-D frames and instance masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _dilate(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        result = np.logical_or.reduce([
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3) for dx in range(3)
        ])
    return result


def _polycam_paths(source_image: Path) -> tuple[Path, Path, Path] | None:
    if source_image.parent.name not in {"corrected_images", "images"}:
        return None
    keyframes = source_image.parent.parent
    camera_group = "corrected_cameras" if source_image.parent.name == "corrected_images" else "cameras"
    paths = (
        keyframes / camera_group / f"{source_image.stem}.json",
        keyframes / "depth" / f"{source_image.stem}.png",
        keyframes / "confidence" / f"{source_image.stem}.png",
    )
    return paths if all(path.is_file() for path in paths) else None


def estimate_polycam_spatial_evidence(
    source_image: Path,
    processed_mask_path: Path,
    rotation_clockwise_degrees: int,
) -> dict | None:
    """Unproject one observation mask into Polycam camera and capture coordinates."""
    paths = _polycam_paths(source_image)
    if paths is None:
        return None
    camera_path, depth_path, confidence_path = paths
    camera = json.loads(camera_path.read_text())
    with Image.open(processed_mask_path) as stored_mask:
        mask_image = stored_mask.convert("L")
    if rotation_clockwise_degrees:
        mask_image = mask_image.rotate(rotation_clockwise_degrees, expand=True)
    with Image.open(depth_path) as stored_depth:
        depth = np.asarray(stored_depth, dtype=np.float64) / 1000.0
    with Image.open(confidence_path) as stored_confidence:
        confidence = np.asarray(stored_confidence)
    mask = np.asarray(
        mask_image.resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)
    ) > 0
    high_confidence = (confidence == confidence.max()) & (confidence > 0)
    valid = mask & (depth > 0) & high_confidence
    confidence_filter = "highest_available"
    if np.count_nonzero(valid) < 3:
        valid = mask & (depth > 0) & (confidence > 0)
        confidence_filter = "nonzero_fallback"
    if np.count_nonzero(valid) < 3:
        return None

    rows, columns = np.nonzero(valid)
    scale_x = camera["width"] / depth.shape[1]
    scale_y = camera["height"] / depth.shape[0]
    pixel_x = (columns + 0.5) * scale_x
    pixel_y = (rows + 0.5) * scale_y
    z = depth[valid]
    points_camera = np.column_stack([
        (pixel_x - camera["cx"]) * z / camera["fx"],
        (pixel_y - camera["cy"]) * z / camera["fy"],
        z,
    ])
    rotation = np.array([
        [camera[f"t_{row}{column}"] for column in range(3)]
        for row in range(3)
    ])
    translation = np.array([camera[f"t_{row}3"] for row in range(3)])
    points_capture = points_camera @ rotation.T + translation
    sample_count = min(128, len(points_capture))
    sample_indices = np.linspace(0, len(points_capture) - 1, sample_count, dtype=int)
    point_samples_capture = points_capture[sample_indices]

    ring = _dilate(mask) & ~mask & (depth > 0) & high_confidence
    if np.count_nonzero(ring) < 3:
        ring = _dilate(mask) & ~mask & (depth > 0) & (confidence > 0)
    surface_depth = float(np.median(depth[ring])) if np.any(ring) else None
    hold_depth = float(np.median(z))
    protrusion = (
        max(0.0, surface_depth - hold_depth) if surface_depth is not None else None
    )
    surface_normal_capture = None
    if np.count_nonzero(ring) >= 3:
        ring_rows, ring_columns = np.nonzero(ring)
        ring_z = depth[ring]
        ring_x = ((ring_columns + 0.5) * scale_x - camera["cx"]) * ring_z / camera["fx"]
        ring_y = ((ring_rows + 0.5) * scale_y - camera["cy"]) * ring_z / camera["fy"]
        ring_points = np.column_stack([ring_x, ring_y, ring_z])
        covariance = np.cov(ring_points - np.mean(ring_points, axis=0), rowvar=False)
        _, vectors = np.linalg.eigh(covariance)
        normal_camera = vectors[:, 0]
        if normal_camera[2] > 0:
            normal_camera = -normal_camera
        normal_capture = rotation @ normal_camera
        surface_normal_capture = normal_capture.round(6).tolist()

    return {
        "method": "polycam_masked_rgbd_unprojection",
        "coordinate_frame": "polycam_capture_unaligned",
        "sample_count": int(len(z)),
        "median_depth_m": round(hold_depth, 6),
        "position_camera_m": np.median(points_camera, axis=0).round(6).tolist(),
        "position_capture_m": np.median(points_capture, axis=0).round(6).tolist(),
        "size_camera_m": np.ptp(points_camera, axis=0).round(6).tolist(),
        "size_capture_m": np.ptp(points_capture, axis=0).round(6).tolist(),
        "point_samples_capture_m": point_samples_capture.round(5).tolist(),
        "surface_normal_capture": surface_normal_capture,
        "neighbor_surface_depth_m": (
            round(surface_depth, 6) if surface_depth is not None else None
        ),
        "protrusion_m": round(protrusion, 6) if protrusion is not None else None,
        "confidence_filter": confidence_filter,
        "confidence_valid_fraction": round(float(np.mean(confidence[mask] > 0)), 6),
        "format_assumptions": [
            "uint16_depth_values_are_millimeters",
            "t_00_through_t_23_map_camera_points_to_capture_space",
        ],
    }
