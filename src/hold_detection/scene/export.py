"""One downstream scene schema for image observations and reconstructed Physical Holds."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from trimesh.transformations import transform_points

from ..capture.images import processed_image
from ..perception.geometry import crop_box
from ..perception.hold_types import summarize_type_assignments
from ..reconstruction.colmap import colmap_depth_path, lift_mask, read_colmap_cameras, read_colmap_depth
from ..reconstruction.orientation import estimate_up, glb_frame_transform, manual_up
from .artifacts import SCENE_FILENAME, SCENE_SCHEMA_VERSION, frame_record_paths


BOTTOM_PERCENTILE = 0.5
VERTICAL_TOLERANCE_DEG = 2.0


def _mean_color(image, mask, bbox):
    if image is None:
        return None, None
    if mask is not None and mask.shape == image.shape[:2] and mask.any():
        return image[mask].mean(axis=0).tolist(), "mask"
    x1, y1, x2, y2 = crop_box(bbox, image.shape[1], image.shape[0], padding=0)
    if x2 <= x1 or y2 <= y1:
        return None, None
    pixels = image[y1:y2, x1:x2].reshape(-1, 3)
    return pixels.mean(axis=0).tolist(), "bbox"


def _observation(frame, hold):
    width, height = frame["processed_size_wh"]
    x1, y1, x2, y2 = hold["bbox_xyxy"]
    return {"id": f"{frame['frame_id']}:{hold['hold_id']}", "frame_id": frame["frame_id"],
            "source_image": frame["source_image"], "object_kind": hold.get("object_kind", "hold"),
            "bbox_xyxy": hold["bbox_xyxy"],
            "position_2d_normalized": [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height)],
            "size_2d_normalized": [(x2 - x1) / width, (y2 - y1) / height],
            "color_rgb": hold.get("median_rgb"), "color_lab": hold.get("color_lab"),
            "mean_color": hold.get("mean_color"), "mean_color_source": hold.get("mean_color_source"),
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


def build_wall_scene(observations, mode, reconstruction=None, annotations=None, orientation=None,
                     *, reconstruction_to_glb=None, wall_bottom_height=None):
    """Build the shared hold contract from saved image and reconstruction evidence."""
    if mode not in {"2d", "3d"}:
        raise ValueError("Mode must be 2d or 3d")
    reconstruction = reconstruction or {}
    orientation = orientation if mode == "3d" else None
    up = orientation["up_vector"] if orientation else None
    scale = reconstruction.get("scale_m_per_unit") if mode == "3d" else None
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Scale must be finite and positive")
    units = "normalized_image" if mode == "2d" else "meters" if scale is not None else "arbitrary"
    up_axis = "y" if up is not None and reconstruction_to_glb is not None else None
    height_reference = "mesh_bottom_percentile" if wall_bottom_height is not None else None
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
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-12 else None
        position_glb = wall_normal = None
        if position is not None and reconstruction_to_glb is not None:
            linear = reconstruction_to_glb[:3, :3]
            position_glb = transform_points([position], reconstruction_to_glb)[0].tolist()
            if normal is not None:
                transformed = linear @ normal
                wall_normal = (transformed / np.linalg.norm(transformed)).tolist()
        relative_height = (position_glb[1] - wall_bottom_height
                           if position_glb is not None and wall_bottom_height is not None else None)
        wall_angle = wall_kind = footprint = None
        if normal is not None and up is not None:
            wall_angle = float(np.degrees(np.arcsin(np.clip(normal @ up, -1, 1))))
            if abs(wall_angle) <= VERTICAL_TOLERANCE_DEG:
                wall_kind = "vertical"
            else:
                wall_kind = "overhang" if wall_angle < 0 else "slab"
            across = np.cross(up, normal)
            across_length = np.linalg.norm(across)
            if scale is not None and across_length > 1e-8:
                across /= across_length
                along = np.cross(normal, across)
                footprint = (np.ptp(points @ np.column_stack((across, along)), axis=0) * scale).tolist()
        protrusions = [item["protrusion"] for item in valid if item["protrusion"] is not None]
        typed = [item for item in group if item["hold_type"] != "unknown"]
        types = {item["hold_type"] for item in typed}
        best = max(typed, key=lambda item: item["hold_type_confidence"] or 0) if len(types) == 1 else None
        first = group[0]
        mask_colors = [item["mean_color"] for item in group if item.get("mean_color_source") == "mask"]
        box_colors = [item["mean_color"] for item in group if item.get("mean_color_source") == "bbox"]
        colors = mask_colors or box_colors
        flags = []
        if mode == "3d":
            flags.append("grip_surface_metrics_unavailable")
            if position is None:
                flags.append("unmapped_3d")
            if scale is None:
                flags.append("metric_scale_unavailable")
            if up is None:
                flags.append("up_direction_unavailable")
        if box_colors and not mask_colors:
            flags.append("bbox_color_fallback")
        hold = {
            "id": identity,
            "position": first["position_2d_normalized"] if mode == "2d" else position_glb,
            "height_m": relative_height if scale is not None else None,
            "facing": None, "wall_normal": wall_normal,
            "wall_angle_deg": wall_angle, "wall_kind": wall_kind,
            "protrusion_m": float(np.median(protrusions) * scale) if protrusions and scale else None,
            "footprint_m": footprint, "grip_area_m2": None, "slope_deg": None,
            "incut_angle_deg": None, "edge_depth_m": None, "two_hands": None,
            "mean_color": np.mean(colors, axis=0).round().astype(int).tolist() if colors else None,
            "is_start": False, "is_top": False, "flags": flags,
            "n_faces": None, "surface_area_m2": None, "wall_like_area_frac": None, "plane_inliers": None,
            "object_kind": first["object_kind"],
            "observation_ids": [item["id"] for item in group], "observation_count": len(group),
            "position_2d_normalized": first["position_2d_normalized"],
            "size_2d_normalized": first["size_2d_normalized"], "reference_frame_id": first["frame_id"],
            "position_3d": position.tolist() if valid else None,
            "position_3d_m": (position * scale).tolist() if valid and scale else None,
            "position_glb": position_glb,
            "height_3d": height, "height_from_wall_bottom": relative_height,
            "size_3d_m": (size * scale).tolist() if valid and scale else None,
            "aabb_volume_m3": float(np.prod(size * scale)) if valid and scale else None,
            "volume_m3": None, "surface_normal": None,
            "wall_normal_reconstruction": normal.tolist() if normal is not None else None,
            "hold_type": best["hold_type"] if best else "unknown",
            "hold_type_confidence": best["hold_type_confidence"] if best else None,
            "hold_type_source": best["hold_type_source"] if best else "not_inferred_or_conflicting",
            "type_candidates": [dict(proposal, observation_id=item["id"])
                                for item in group for proposal in item.get("type_proposals", [])],
            "opening_direction": "unknown",
            "color_rgb": first["color_rgb"],
            "route_ids": [], "is_finish": False,
            "geometry_reliable": False,
            "geometry_status": "multi_view" if len(group) > 1 and valid else "single_view" if valid else "unmapped",
            "needs_review": True, "mesh_correspondence": None, "ground_truth_match": None,
        }
        # Human edits are explicit and never overwrite measured coordinates.
        edits = (annotations or {}).get(identity, {})
        for key in ["hold_type", "opening_direction", "route_ids", "is_start", "is_finish"]:
            if key in edits:
                hold[key] = edits[key]
        if "is_top" in edits and "is_finish" in edits and edits["is_top"] != edits["is_finish"]:
            raise ValueError(f"Conflicting is_top and is_finish annotations for {identity}")
        hold["is_top"] = hold["is_finish"] = edits.get("is_top", hold["is_finish"])
        if "hold_type" in edits:
            hold["hold_type_source"], hold["hold_type_confidence"] = "human", None
        holds.append(hold)
    top_holds = [hold["id"] for hold in holds if hold["is_top"]]
    return {"schema_version": SCENE_SCHEMA_VERSION, "units": units, "up_axis": up_axis,
            "source_file": None, "wall_object": None,
            "wall_bottom_height": wall_bottom_height, "wall_bottom_raw_min": None,
            "height_reference": height_reference,
            "params": {"bottom_percentile": BOTTOM_PERCENTILE, "vertical_tol": VERTICAL_TOLERANCE_DEG},
            "holds": holds, "start_hold_ids": [hold["id"] for hold in holds if hold["is_start"]],
            "top_hold_id": top_holds[0] if len(top_holds) == 1 else None,
            "analysis_mode": mode,
            "wall": {"coordinate_frame": "colmap_world" if mode == "3d" else "processed_image",
                     "units": units,
                     "scale_m_per_unit": scale, "up_axis": None,
                     "up_vector": up, "up_orientation": orientation,
                     "height_reference": "reconstruction_origin" if up is not None else None,
                     "glb_path": reconstruction.get("glb_path"), "glb_up_axis": up_axis,
                     "reconstruction_to_glb": (reconstruction_to_glb.tolist()
                                               if reconstruction_to_glb is not None else None),
                     "reconstruction": reconstruction},
            "observations": observations,
            "review": {"unknown_type_count": sum(h["hold_type"] == "unknown" for h in holds),
                       "unmapped_hold_count": sum(h["position_3d"] is None for h in holds) if mode == "3d" else 0},
            "limitations": ["joint_type_scores_are_not_calibrated_probabilities",
                            "occluded_grip_metrics_unavailable", "aabb_volume_is_not_true_hold_volume"]}


def export_wall_scene(run, mode, reconstruction_path=None, annotations_path=None, up_reference_path=None):
    """Lift saved masks, retain all observations and write the graph algorithm's scene.json."""
    run = Path(run).resolve()
    if mode == "2d" and reconstruction_path:
        raise ValueError("Choose 3D mode to export a reconstruction")
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
    frame_paths = frame_record_paths(run)
    summary = json.loads((run / "inference/summary.json").read_text())
    if not frame_paths or summary["frame_count"] != len(frame_paths):
        raise ValueError("Inference frames are incomplete; resume segmentation before export")
    observations = []
    source_file = None
    for frame_path in frame_paths:
        frame = json.loads(frame_path.read_text())
        source = Path(frame["source_image"])
        if mode == "2d":
            source_file = source.name
        pixels = np.asarray(processed_image(frame)) if frame["holds"] and source.is_file() else None
        camera = cameras.get(source.name)
        depth = None
        if camera:
            depth_file = colmap_depth_path(reconstruction["depth_directory"], source.name)
            if depth_file.exists():
                depth = read_colmap_depth(depth_file)
        for hold in frame["holds"]:
            hold = dict(hold)
            mask = None
            if hold.get("mask_path") and (pixels is not None or depth is not None):
                try:
                    with Image.open(frame_path.parent / hold["mask_path"]) as stored:
                        mask = np.asarray(stored.convert("L")) > 0
                except FileNotFoundError:
                    hold["mask_path"] = None
            hold["mean_color"], hold["mean_color_source"] = _mean_color(pixels, mask, hold["bbox_xyxy"])
            if depth is not None and mask is not None:
                if frame.get("rotation_clockwise_degrees", 0):
                    raise ValueError("COLMAP masks must use unrotated undistorted images")
                hold["geometry"] = lift_mask(depth, mask, camera)
                if hold["geometry"]:
                    hold["geometry"]["depth_map_type"] = depth_file.name.rsplit(".", 2)[1]
            observations.append(_observation(frame, hold))
    annotations = json.loads(Path(annotations_path).read_text()) if annotations_path else None
    transform = upright = mesh = None
    wall_bottom = wall_bottom_raw_min = None
    scale = reconstruction.get("scale_m_per_unit")
    glb_path = reconstruction.get("glb_path")
    if reconstruction:
        import trimesh

        mesh = trimesh.load(reconstruction["mesh_path"], force="mesh", process=False)
        up = orientation["up_vector"] if orientation is not None else None
        transform = glb_frame_transform(up, cameras, scale)
        if up is not None:
            upright = run / "wall_upright.glb"
            glb_path = str(upright)
            heights = mesh.vertices @ transform[1, :3] + transform[1, 3]
            wall_bottom = float(np.percentile(heights, BOTTOM_PERCENTILE))
            wall_bottom_raw_min = float(heights.min())
        source_file = Path(glb_path).name if glb_path else None

    scene = build_wall_scene(observations, mode, reconstruction, annotations, orientation,
                             reconstruction_to_glb=transform if glb_path else None,
                             wall_bottom_height=wall_bottom)
    scene.update(source_file=source_file, wall_bottom_raw_min=wall_bottom_raw_min)
    scene["wall"]["input_crop"] = summary.get("input_crop")
    scene["wall"]["glb_path"] = glb_path
    if summary.get("grip_type_inference_enabled"):
        minimum_score = summary.get("settings", {}).get("type_min_score", 0.5)
        scene["review"]["type_assignment"] = summarize_type_assignments(observations, minimum_score)
    if mesh is not None:
        if glb_path:
            # trimesh uses the PLY node name when exporting a single mesh to GLB.
            scene["wall_object"] = next(iter(mesh.scene().graph.nodes_geometry), None)
        tree = cKDTree(np.asarray(mesh.vertices))
        for hold in scene["holds"]:
            if hold["position_3d"] is None:
                continue
            distance, index = tree.query(hold["position_3d"])
            anchor = mesh.vertices[index]
            hold["mesh_correspondence"] = {
                "method": "nearest_mesh_vertex", "mesh_path": reconstruction["mesh_path"],
                "vertex_index": int(index), "position_3d": anchor.tolist(),
                "position_glb": transform_points([anchor], transform)[0].tolist() if glb_path else None,
                "distance_reconstruction_units": float(distance),
                "distance_m": float(distance * scale) if scale else None,
                "is_semantic_hold_object_match": False, "needs_review": True,
            }
        if upright is not None:
            mesh.apply_transform(transform)
            with TemporaryDirectory(dir=run, prefix=".glb-") as directory:
                temporary = Path(directory) / upright.name
                mesh.export(temporary)
                temporary.replace(upright)
    for observation in scene["observations"]:
        if observation["mask_path"]:
            observation["mask_path"] = "inference/" + observation["mask_path"]
    (run / SCENE_FILENAME).write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    return scene
