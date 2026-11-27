"""Build route-centric scene artifacts from frame-local hold observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..perception.color import assign_color_groups
from ..review.gallery import write_color_overlay


def _point_overlap(first: dict, second: dict, tolerance_m: float) -> float:
    first_points = np.asarray(first.get("point_samples_capture_m", []), dtype=float)
    second_points = np.asarray(second.get("point_samples_capture_m", []), dtype=float)
    if not len(first_points) or not len(second_points):
        return 0.0
    distances = np.linalg.norm(
        first_points[:, None, :] - second_points[None, :, :], axis=2
    )
    first_coverage = float(np.mean(np.min(distances, axis=1) <= tolerance_m))
    second_coverage = float(np.mean(np.min(distances, axis=0) <= tolerance_m))
    return min(first_coverage, second_coverage)


def _fuse_spatial_observations(
    observations: list[dict], distance_m: float, overlap_tolerance_m: float = 0.04,
    minimum_overlap: float = 0.12,
) -> list[dict]:
    spatial = [item for item in observations if item.get("spatial_evidence")]
    spatial.sort(key=lambda item: (
        item.get("route_candidate_id") or "",
        *item["spatial_evidence"]["position_capture_m"],
        item["observation_id"],
    ))
    groups: list[list[dict]] = []
    for observation in spatial:
        position = np.asarray(observation["spatial_evidence"]["position_capture_m"])
        matching = None
        for group in groups:
            if group[0].get("route_candidate_id") != observation.get("route_candidate_id"):
                continue
            if group[0]["object_kind"] != observation["object_kind"]:
                continue
            if any(member["frame_id"] == observation["frame_id"] for member in group):
                continue
            center = np.median([
                member["spatial_evidence"]["position_capture_m"] for member in group
            ], axis=0)
            center_distance = float(np.linalg.norm(position - center))
            overlap = max(
                _point_overlap(
                    observation["spatial_evidence"], member["spatial_evidence"],
                    overlap_tolerance_m,
                )
                for member in group
            )
            if center_distance <= distance_m and (
                overlap >= minimum_overlap or center_distance <= overlap_tolerance_m
            ):
                matching = group
                break
        if matching is None:
            groups.append([observation])
        else:
            matching.append(observation)

    groups.sort(key=lambda group: tuple(np.median([
        member["spatial_evidence"]["position_capture_m"] for member in group
    ], axis=0)))
    physical_holds = []
    confirmed_groups = [
        group for group in groups if len({item["frame_id"] for item in group}) >= 2
    ]
    for index, group in enumerate(confirmed_groups, 1):
        evidence = [member["spatial_evidence"] for member in group]
        position = np.median([item["position_capture_m"] for item in evidence], axis=0)
        size = np.median([item["size_capture_m"] for item in evidence], axis=0)
        normals = [item["surface_normal_capture"] for item in evidence
                   if item["surface_normal_capture"] is not None]
        normal = np.median(normals, axis=0) if normals else None
        if normal is not None and np.linalg.norm(normal):
            normal = normal / np.linalg.norm(normal)
        protrusions = [item["protrusion_m"] for item in evidence if item["protrusion_m"] is not None]
        aabb_volume = float(np.prod(np.maximum(size, 0.0)))
        median_protrusion = float(np.median(protrusions)) if protrusions else None
        footprint_extents = np.sort(np.maximum(size, 0.0))[-2:]
        protrusion_prism_volume = (
            float(np.prod(footprint_extents) * median_protrusion)
            if median_protrusion is not None else None
        )
        physical_holds.append({
            "physical_hold_id": f"physical_hold_{index:04d}",
            "route_candidate_id": group[0].get("route_candidate_id"),
            "object_kind": group[0]["object_kind"],
            "coordinate_frame": "polycam_capture_unaligned",
            "position_3d_m": position.round(6).tolist(),
            "estimated_size_m": size.round(6).tolist(),
            "aabb_volume_m3": round(aabb_volume, 9),
            "protrusion_prism_volume_m3": (
                round(protrusion_prism_volume, 9)
                if protrusion_prism_volume is not None else None
            ),
            "surface_normal": (
                normal.round(6).tolist() if normal is not None else None
            ),
            "protrusion_m": (
                round(median_protrusion, 6) if median_protrusion is not None else None
            ),
            "observation_ids": [item["observation_id"] for item in group],
            "observation_count": len(group),
            "fusion_method": "route_color_and_3d_point_overlap",
            "fusion_distance_m": distance_m,
            "point_overlap_tolerance_m": overlap_tolerance_m,
            "minimum_point_overlap": minimum_overlap,
            "needs_review": True,
        })
    return physical_holds


def build_scene(run: Path, color_distance_threshold: float = 22.0) -> dict:
    """Create the draft planner scene and route review images for an inference run."""
    frame_paths = sorted(run.glob("*/holds.json"))
    frames: list[tuple[Path, dict]] = [
        (path, json.loads(path.read_text())) for path in frame_paths
    ]
    from .spatial import estimate_polycam_spatial_evidence

    route_candidates = assign_color_groups([frame for _, frame in frames], color_distance_threshold)
    spatial_observation_count = 0
    for frame_path, frame in frames:
        for hold in frame["holds"]:
            hold["spatial_evidence"] = None
            if hold.get("mask_path"):
                hold["spatial_evidence"] = estimate_polycam_spatial_evidence(
                    Path(frame["source_image"]),
                    frame_path.parent / hold["mask_path"],
                    frame.get("rotation_clockwise_degrees", 0),
                )
                spatial_observation_count += int(hold["spatial_evidence"] is not None)

    scene_frames = []
    scene_observations = []
    for frame_path, frame in frames:
        frame_path.write_text(json.dumps(frame, indent=2, allow_nan=False) + "\n")
        write_color_overlay(frame_path, frame)
        hold_observations = [
            {
                "observation_id": f"{frame['frame_id']}:{hold['hold_id']}",
                "frame_id": frame["frame_id"],
                "hold_id": hold["hold_id"],
                "object_kind": hold["object_kind"],
                "bbox_xyxy": hold["bbox_xyxy"],
                "mask_path": (
                    f"{frame_path.parent.name}/{hold['mask_path']}"
                    if hold.get("mask_path") else None
                ),
                "route_candidate_id": hold.get("route_candidate_id"),
                "median_rgb": hold.get("median_rgb"),
                "color_lab": hold.get("color_lab"),
                "color_method": hold.get("color_method"),
                "detection_confidence": hold["detector_score"],
                "segmentation_confidence": hold.get("segmentation_score"),
                "spatial_evidence": hold.get("spatial_evidence"),
                "needs_review": hold["needs_review"],
            }
            for hold in frame["holds"]
        ]
        scene_observations.extend(hold_observations)
        scene_frames.append({
            "frame_id": frame["frame_id"],
            "source_image": frame["source_image"],
            "processed_size_wh": frame["processed_size_wh"],
            "hold_observations": hold_observations,
        })

    fusion_distance_m = 0.12
    physical_holds = _fuse_spatial_observations(scene_observations, fusion_distance_m)
    glb_candidates = set()
    capture_roots = set()
    for _, frame in frames:
        source = Path(frame["source_image"])
        if source.parent.name in {"corrected_images", "images"}:
            capture_root = source.parent.parent.parent
            capture_roots.add(capture_root)
            glb_candidates.update(capture_root.parent.glob("*.glb"))
    glb_path = str(next(iter(glb_candidates)).resolve()) if len(glb_candidates) == 1 else None
    registration = None
    if spatial_observation_count and glb_path and len(capture_roots) == 1:
        try:
            from .registration import (
                load_glb_vertices, match_points_to_glb, register_capture_to_glb,
                transform_points,
            )

            registration = register_capture_to_glb(
                next(iter(capture_roots)), Path(glb_path)
            )
            if registration["status"] == "registered":
                glb_vertices = load_glb_vertices(Path(glb_path))
                observations_by_id = {
                    item["observation_id"]: item for item in scene_observations
                }
                matrix = np.asarray(
                    registration["capture_to_glb_matrix_row_major"], dtype=float
                )
                for physical_hold in physical_holds:
                    position = np.asarray(physical_hold["position_3d_m"], dtype=float)
                    physical_hold["position_glb_m"] = transform_points(
                        position[None, :], registration
                    )[0].round(6).tolist()
                    normal = physical_hold.get("surface_normal")
                    physical_hold["surface_normal_glb"] = (
                        (np.asarray(normal) @ matrix[:3, :3].T).round(6).tolist()
                        if normal is not None else None
                    )
                    samples = []
                    for observation_id in physical_hold["observation_ids"]:
                        evidence = observations_by_id[observation_id]["spatial_evidence"]
                        samples.extend(evidence.get("point_samples_capture_m", []))
                    physical_hold["glb_match"] = match_points_to_glb(
                        np.asarray(samples, dtype=float), Path(glb_path), registration,
                        glb_vertices,
                    )
                    physical_hold["needs_review"] |= physical_hold["glb_match"]["needs_review"]
        except (OSError, RuntimeError, ValueError) as error:
            registration = {
                "status": "failed", "error": str(error),
                "method": "pca_initialized_trimmed_point_to_point_icp",
            }

    scene = {
        "schema_version": "2.0-draft",
        "wall_capture_id": run.name,
        "coordinate_space": "image",
        "route_candidate_assumption": "color_groups_are_candidates_not_unique_routes",
        "color_distance_threshold": color_distance_threshold,
        "route_candidates": route_candidates,
        "frames": scene_frames,
        "physical_holds": physical_holds,
        "reconstruction": {
            "status": ("multi_view_rgbd_fusion" if physical_holds else
                       "rgbd_spatial_evidence" if spatial_observation_count else "not_run"),
            "spatial_observation_count": spatial_observation_count,
            "coordinate_frame": (
                "polycam_capture_unaligned" if spatial_observation_count else None
            ),
            "glb_registration": (
                registration["status"] if registration else "not_registered"
            ),
            "glb_registration_details": registration,
            "glb_path": glb_path,
            "fusion_distance_m": fusion_distance_m,
        },
    }
    (run / "scene.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    return scene
