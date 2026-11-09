"""One downstream scene schema for image observations and reconstructed Physical Holds."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from .reconstruction import lift_mask, read_colmap_cameras, read_colmap_depth
from .wall_orientation import estimate_up, manual_up


def assign_type_proposals(holds, proposals, minimum_score=0.5, minimum_iou=0.3):
    """Match detector boxes one-to-one; retain segmentation instances without a type match."""
    from .geometry import box_iou
    from scipy.optimize import linear_sum_assignment
    result = [dict(hold, hold_type="unknown", hold_type_confidence=None,
                   hold_type_source="not_inferred", type_needs_review=True, type_proposals=[],
                   needs_review=True,
                   review_reasons=sorted(set(hold.get("review_reasons", [])) | {"rgb_type_requires_review"}))
              for hold in holds]
    if not holds or not proposals:
        return result
    overlap = np.array([[box_iou(h["bbox_xyxy"], p["bbox_xyxy"]) for p in proposals] for h in holds])
    for row, hold in enumerate(result):
        hold["type_proposals"] = [{"label": proposal["label"], "score": proposal["score"],
                                   "iou": float(overlap[row, column])}
                                  for column, proposal in enumerate(proposals) if overlap[row, column] >= minimum_iou]
    # Low-score boxes must not steal the assignment from a viable proposal.
    valid = overlap * np.array([p["score"] >= minimum_score for p in proposals])
    for row, column in zip(*linear_sum_assignment(-valid)):
        proposal = proposals[column]
        if valid[row, column] < minimum_iou:
            continue
        result[row].update(hold_type=proposal["label"], hold_type_confidence=proposal["score"],
                           hold_type_source="yolo26_rgb_joint_score", type_needs_review=True,
                           type_match_iou=float(overlap[row, column]))
    return result


def _observation(frame, hold):
    width, height = frame["processed_size_wh"]
    x1, y1, x2, y2 = hold["bbox_xyxy"]
    return {"id": f"{frame['frame_id']}:{hold['hold_id']}", "frame_id": frame["frame_id"],
            "source_image": frame["source_image"], "object_kind": hold.get("object_kind", "hold"),
            "bbox_xyxy": hold["bbox_xyxy"],
            "position_2d_normalized": [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height)],
            "size_2d_normalized": [(x2 - x1) / width, (y2 - y1) / height],
            "color_rgb": hold.get("median_rgb"), "color_lab": hold.get("color_lab"),
            "color_group_id": hold.get("route_candidate_id"),
            "mask_path": f"{frame['frame_id']}/{hold['mask_path']}" if hold.get("mask_path") else None,
            "hold_type": hold.get("hold_type", "unknown"),
            "hold_type_confidence": hold.get("hold_type_confidence"),
            "hold_type_source": hold.get("hold_type_source", "not_inferred"),
            "type_proposals": hold.get("type_proposals", []),
            "detection_confidence": hold["detector_score"], "geometry": hold.get("geometry")}


def fuse_holds(observations, scale_m_per_unit=None):
    """Fuse overlapping visible 3D evidence; preserve single-view and unmapped holds for review."""
    groups = []
    for item in sorted(observations, key=lambda h: h["id"]):
        geometry = item.get("geometry")
        match = None
        if geometry:
            for group in groups:
                first = group[0]
                if not first.get("geometry") or first["object_kind"] != item["object_kind"]:
                    continue
                if any(member["frame_id"] == item["frame_id"] for member in group):
                    continue
                if item.get("color_lab") and first.get("color_lab"):
                    if np.linalg.norm(np.array(item["color_lab"]) - first["color_lab"]) > 30:
                        continue
                tolerance = 0.04 / scale_m_per_unit if scale_m_per_unit else max(
                    1e-8, float(np.linalg.norm(geometry["extent"])) * 0.15)
                samples = np.asarray(geometry["samples"])
                overlaps = []
                for member in group:
                    other = np.asarray(member["geometry"]["samples"])
                    overlap = min(np.mean(cKDTree(other).query(samples)[0] <= tolerance),
                                  np.mean(cKDTree(samples).query(other)[0] <= tolerance))
                    overlaps.append(overlap)
                if min(overlaps) >= 0.15:
                    match = group
                    break
        if match is None:
            groups.append([item])
        else:
            match.append(item)
    return groups


def build_wall_scene(observations, mode, reconstruction=None, annotations=None, orientation=None):
    """Keep identical keys in 2D and 3D; unavailable geometry is null, never fabricated."""
    if mode not in {"2d", "3d"}:
        raise ValueError("Mode must be 2d or 3d")
    reconstruction = reconstruction or {}
    orientation = orientation if mode == "3d" else None
    up = orientation["up_vector"] if orientation else None
    scale = reconstruction.get("scale_m_per_unit") if mode == "3d" else None
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Scale must be finite and positive")
    groups = fuse_holds(observations, scale) if mode == "3d" else [[item] for item in observations]
    holds = []
    for group in groups:
        identity = "hold_" + hashlib.sha256(group[0]["id"].encode()).hexdigest()[:12]
        valid = [item["geometry"] for item in group if item.get("geometry")] if mode == "3d" else []
        points = np.concatenate([item["samples"] for item in valid]) if valid else None
        position = np.median(points, axis=0) if valid else None
        height = float(position @ up) if valid and up is not None else None
        size = np.ptp(points, axis=0) if valid else None
        normals = [item["wall_normal"] for item in valid if item["wall_normal"] is not None]
        normal = np.mean(normals, axis=0) if normals else None
        if normal is not None:
            normal /= max(1e-12, np.linalg.norm(normal))
        protrusions = [item["protrusion"] for item in valid if item["protrusion"] is not None]
        typed = [item for item in group if item["hold_type"] != "unknown"]
        types = {item["hold_type"] for item in typed}
        best = max(typed, key=lambda item: item["hold_type_confidence"] or 0) if len(types) == 1 else None
        first = group[0]
        hold = {
            "id": identity, "object_kind": first["object_kind"],
            "observation_ids": [item["id"] for item in group], "observation_count": len(group),
            "position_2d_normalized": first["position_2d_normalized"],
            "size_2d_normalized": first["size_2d_normalized"], "reference_frame_id": first["frame_id"],
            "position_3d": position.tolist() if valid else None,
            "position_3d_m": (position * scale).tolist() if valid and scale else None,
            "height_3d": height,
            "height_m": height * scale if height is not None and scale is not None else None,
            "size_3d_m": (size * scale).tolist() if valid and scale else None,
            "protrusion_m": float(np.median(protrusions) * scale) if protrusions and scale else None,
            "aabb_volume_m3": float(np.prod(size * scale)) if valid and scale else None,
            "volume_m3": None, "surface_normal": None,
            "wall_normal": normal.tolist() if normal is not None else None,
            "hold_type": best["hold_type"] if best else "unknown",
            "hold_type_confidence": best["hold_type_confidence"] if best else None,
            "hold_type_source": best["hold_type_source"] if best else "not_inferred_or_conflicting",
            "type_candidates": [dict(proposal, observation_id=item["id"])
                                for item in group for proposal in item.get("type_proposals", [])],
            "opening_direction": "unknown", "grip_area_m2": None, "edge_depth_m": None,
            "color_rgb": first["color_rgb"], "color_group_id": first["color_group_id"],
            "route_ids": [], "is_start": False, "is_finish": False,
            "geometry_reliable": False,
            "geometry_status": "multi_view" if len(group) > 1 and valid else "single_view" if valid else "unmapped",
            "needs_review": True, "mesh_correspondence": None, "ground_truth_match": None,
        }
        # Human edits are explicit and never overwrite measured coordinates.
        edits = (annotations or {}).get(identity, {})
        for key in ["hold_type", "opening_direction", "route_ids", "is_start", "is_finish"]:
            if key in edits:
                hold[key] = edits[key]
        if "hold_type" in edits:
            hold["hold_type_source"], hold["hold_type_confidence"] = "human", None
        holds.append(hold)
    return {"schema_version": "3.0", "analysis_mode": mode,
            "wall": {"coordinate_frame": "colmap_world" if mode == "3d" else "processed_image",
                     "units": reconstruction.get("units", "arbitrary" if mode == "3d" else "normalized_image"),
                     "scale_m_per_unit": scale, "up_axis": None,
                     "up_vector": up, "up_orientation": orientation,
                     "height_reference": "reconstruction_origin" if up is not None else None,
                     "glb_path": reconstruction.get("glb_path"), "reconstruction": reconstruction},
            "holds": holds, "observations": observations,
            "review": {"unknown_type_count": sum(h["hold_type"] == "unknown" for h in holds),
                       "unmapped_hold_count": sum(h["position_3d"] is None for h in holds) if mode == "3d" else 0},
            "limitations": ["color_groups_are_not_unique_routes", "joint_type_scores_are_not_calibrated_probabilities",
                            "occluded_grip_metrics_unavailable", "aabb_volume_is_not_true_hold_volume"]}


def export_wall_scene(run, mode, reconstruction_path=None, annotations_path=None, up_reference_path=None):
    """Lift saved masks, retain all observations and write the graph algorithm's scene.json."""
    run = Path(run).resolve()
    reconstruction = json.loads(Path(reconstruction_path).read_text()) if reconstruction_path else {}
    if mode == "3d" and not reconstruction:
        raise ValueError("3D export requires reconstruction.json")
    if up_reference_path and mode != "3d":
        raise ValueError("An up reference requires a 3D export")
    cameras = read_colmap_cameras(reconstruction["camera_model"]) if reconstruction else {}
    orientation = None
    if mode == "3d":
        orientation = (manual_up(cameras, reconstruction["depth_directory"],
                                 json.loads(Path(up_reference_path).read_text()))
                       if up_reference_path else estimate_up(cameras))
    frame_paths = sorted((run / "inference").glob("*/holds.json"))
    summary = json.loads((run / "inference/summary.json").read_text())
    if not frame_paths or summary["frame_count"] != len(frame_paths):
        raise ValueError("Inference frames are incomplete; resume segmentation before export")
    observations = []
    for frame_path in frame_paths:
        frame = json.loads(frame_path.read_text())
        camera = cameras.get(Path(frame["source_image"]).name)
        depth = None
        if camera:
            name = Path(frame["source_image"]).name
            depth_file = Path(reconstruction["depth_directory"]) / f"{name}.geometric.bin"
            if depth_file.exists():
                depth = read_colmap_depth(depth_file)
        for hold in frame["holds"]:
            hold = dict(hold)
            if depth is not None and hold.get("mask_path"):
                with Image.open(frame_path.parent / hold["mask_path"]) as stored:
                    mask = np.asarray(stored.convert("L")) > 0
                if frame.get("rotation_clockwise_degrees", 0):
                    raise ValueError("COLMAP masks must use unrotated undistorted images")
                hold["geometry"] = lift_mask(depth, mask, camera)
            observations.append(_observation(frame, hold))
    annotations = json.loads(Path(annotations_path).read_text()) if annotations_path else None
    scene = build_wall_scene(observations, mode, reconstruction, annotations, orientation)
    scene["wall"]["input_crop"] = summary.get("input_crop")
    if reconstruction:
        import trimesh
        mesh = trimesh.load(reconstruction["mesh_path"], force="mesh", process=False)
        tree = cKDTree(np.asarray(mesh.vertices))
        scale = reconstruction.get("scale_m_per_unit")
        for hold in scene["holds"]:
            if hold["position_3d"] is None:
                continue
            distance, index = tree.query(hold["position_3d"])
            hold["mesh_correspondence"] = {
                "method": "nearest_mesh_vertex", "mesh_path": reconstruction["mesh_path"],
                "vertex_index": int(index), "position_3d": mesh.vertices[index].tolist(),
                "distance_reconstruction_units": float(distance),
                "distance_m": float(distance * scale) if scale else None,
                "is_semantic_hold_object_match": False, "needs_review": True,
            }
    for observation in scene["observations"]:
        if observation["mask_path"]:
            observation["mask_path"] = "inference/" + observation["mask_path"]
    (run / "scene.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    # Keep the previous notebook filename as a byte-identical compatibility artifact.
    (run / "wall_scene.json").write_text((run / "scene.json").read_text())
    return scene
