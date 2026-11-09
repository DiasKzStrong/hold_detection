"""Estimate vertical in the COLMAP frame or calibrate it from an image reference."""

import numpy as np

from .reconstruction import colmap_depth_path, read_colmap_depth, reference_camera_points


def estimate_up(cameras):
    """Fit an up vector perpendicular to camera-right axes in upright views."""
    result = {"status": "manual_required", "source": "camera_poses",
              "up_vector": None, "camera_count": len(cameras)}
    if len(cameras) < 3:
        return dict(result, reason="too_few_registered_views")

    # Rows of COLMAP's world-to-camera R express the camera axes in world coordinates.
    rotations = np.asarray([camera["rotation"] for camera in cameras.values()])
    right, image_up = rotations[:, 0], -rotations[:, 1]
    _, singular, axes = np.linalg.svd(right, full_matrices=False)
    spread = singular[1] / np.sqrt(len(cameras))
    result["camera_right_spread"] = float(spread)
    if spread < 0.12:
        return dict(result, reason="camera_angles_too_similar")

    up = axes[-1]
    if np.median(image_up @ up) < 0:
        up = -up
    roll_error = np.degrees(np.arcsin(np.clip(np.abs(right @ up), 0, 1)))
    result["roll_error_p80_deg"] = float(np.quantile(roll_error, 0.8))
    result["upright_fraction"] = float(np.mean(image_up @ up > 0.15))
    if (singular[2] / singular[1] > 0.45 or result["roll_error_p80_deg"] > 8
            or result["upright_fraction"] < 0.8):
        return dict(result, reason="camera_orientations_disagree")
    return dict(result, status="estimated", up_vector=up.tolist())


def manual_up(cameras, depth_directory, reference):
    """Lift the bottom and top of a known vertical line and use their signed direction."""
    if not isinstance(reference, dict) or not {"image", "bottom_px", "top_px"} <= reference.keys():
        raise ValueError("Up reference requires image, bottom_px, and top_px")
    image = reference["image"]
    if not isinstance(image, str) or image not in cameras:
        raise ValueError("Choose an undistorted image registered by COLMAP")
    depth_path = colmap_depth_path(depth_directory, image)
    if not depth_path.is_file():
        raise ValueError("This image has no stereo depth; choose another frame")
    camera = cameras[image]
    depth = read_colmap_depth(depth_path)
    bottom, top = reference_camera_points(
        depth, camera, [reference["bottom_px"], reference["top_px"]], require_support=True,
    )
    delta = (top - bottom) @ np.asarray(camera["rotation"])
    length = np.linalg.norm(delta)
    pixel_gap = np.linalg.norm(np.subtract(
        np.asarray(reference["top_px"], dtype=float), np.asarray(reference["bottom_px"], dtype=float),
    ))
    if length <= 1e-9 or pixel_gap < 5:
        raise ValueError("Pick two separated points on the same vertical line")
    return {"status": "calibrated", "source": "manual_reference",
            "up_vector": (delta / length).tolist(), "reference": reference,
            "depth_map_type": depth_path.name.rsplit(".", 2)[1]}
