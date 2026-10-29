"""Build route-centric scene artifacts from frame-local hold observations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROUTE_COLORS = (
    "#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
)


def rgb_to_lab(rgb: list[int]) -> list[float]:
    """Convert an sRGB triplet to CIE Lab under the D65 illuminant."""
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92,
                      ((values + 0.055) / 1.055) ** 2.4)
    xyz = np.array([
        0.4124564 * linear[0] + 0.3575761 * linear[1] + 0.1804375 * linear[2],
        0.2126729 * linear[0] + 0.7151522 * linear[1] + 0.0721750 * linear[2],
        0.0193339 * linear[0] + 0.1191920 * linear[1] + 0.9503041 * linear[2],
    ]) / np.array([0.95047, 1.0, 1.08883])
    pivot = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    lab = np.array([
        116 * pivot[1] - 16,
        500 * (pivot[0] - pivot[1]),
        200 * (pivot[1] - pivot[2]),
    ])
    return lab.round(3).tolist()


@dataclass
class _ColorCluster:
    members: list[dict] = field(default_factory=list)
    center: np.ndarray | None = None

    def add(self, observation: dict) -> None:
        self.members.append(observation)
        values = np.asarray([member["color_lab"] for member in self.members])
        self.center = np.median(values, axis=0)


def _cluster_route_candidates(observations: list[dict], threshold: float) -> list[_ColorCluster]:
    clusters: list[_ColorCluster] = []
    ordered = sorted(observations, key=lambda item: tuple(item["color_lab"]))
    for observation in ordered:
        value = np.asarray(observation["color_lab"])
        distances = [float(np.linalg.norm(value - cluster.center)) for cluster in clusters]
        nearest = int(np.argmin(distances)) if distances else None
        if nearest is None or distances[nearest] > threshold:
            cluster = _ColorCluster()
            cluster.add(observation)
            clusters.append(cluster)
        else:
            clusters[nearest].add(observation)
    clusters.sort(key=lambda cluster: tuple(cluster.center.round(6)))
    for index, cluster in enumerate(clusters, 1):
        candidate_id = f"route_candidate_{index:04d}"
        for observation in cluster.members:
            observation["route_candidate_id"] = candidate_id
    return clusters


def _render_routes(run: Path, frame_path: Path, frame: dict) -> None:
    with Image.open(frame["source_image"]) as source:
        image = source.convert("RGB")
    rotation = frame.get("rotation_clockwise_degrees", 0)
    if rotation:
        image = image.rotate(-rotation, expand=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for hold in frame["holds"]:
        candidate_id = hold.get("route_candidate_id")
        if not candidate_id:
            continue
        route_number = int(candidate_id.rsplit("_", 1)[1])
        color = ROUTE_COLORS[(route_number - 1) % len(ROUTE_COLORS)]
        mask_path = hold.get("mask_path")
        if mask_path:
            with Image.open(frame_path.parent / mask_path) as stored_mask:
                mask = stored_mask.convert("L")
            tint = Image.new("RGB", image.size, color)
            alpha = mask.point(lambda value: 78 if value else 0)
            overlay = Image.composite(tint, overlay, alpha)
            draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = hold["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f"{hold['hold_id']}  {candidate_id}"
        anchor = (x1, max(0, y1 - 14))
        bounds = draw.textbbox(anchor, label)
        draw.rectangle(bounds, fill="black")
        draw.text(anchor, label, fill=color)
    overlay.save(frame_path.parent / "routes.jpg", quality=95)


def _fuse_spatial_observations(observations: list[dict], distance_m: float) -> list[dict]:
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
            if any(member["frame_id"] == observation["frame_id"] for member in group):
                continue
            center = np.median([
                member["spatial_evidence"]["position_capture_m"] for member in group
            ], axis=0)
            if float(np.linalg.norm(position - center)) <= distance_m:
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
        physical_holds.append({
            "physical_hold_id": f"physical_hold_{index:04d}",
            "route_candidate_id": group[0].get("route_candidate_id"),
            "object_kind": group[0]["object_kind"],
            "coordinate_frame": "polycam_capture_unaligned",
            "position_3d_m": position.round(6).tolist(),
            "estimated_size_m": size.round(6).tolist(),
            "surface_normal": (
                normal.round(6).tolist() if normal is not None else None
            ),
            "protrusion_m": (
                round(float(np.median(protrusions)), 6) if protrusions else None
            ),
            "observation_ids": [item["observation_id"] for item in group],
            "observation_count": len(group),
            "fusion_method": "route_color_and_capture_position_radius",
            "fusion_distance_m": distance_m,
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

    observations = []
    spatial_observation_count = 0
    for frame_path, frame in frames:
        with Image.open(frame["source_image"]) as source:
            processed = source.convert("RGB")
        rotation = frame.get("rotation_clockwise_degrees", 0)
        if rotation:
            processed = processed.rotate(-rotation, expand=True)
        for hold in frame["holds"]:
            rgb = hold.get("median_rgb")
            if rgb is None:
                x1, y1, x2, y2 = hold["bbox_xyxy"]
                bounds = (
                    max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
                    min(processed.width, int(np.ceil(x2))),
                    min(processed.height, int(np.ceil(y2))),
                )
                crop = np.asarray(processed.crop(bounds))
                rgb = (
                    np.median(crop.reshape(-1, 3), axis=0).round().astype(int).tolist()
                    if crop.size else None
                )
                hold["median_rgb"] = rgb
                hold["color_method"] = "bbox_median_rgb_fallback"
                hold["needs_review"] = True
                if "bbox_color_fallback" not in hold["review_reasons"]:
                    hold["review_reasons"].append("bbox_color_fallback")
            hold["color_lab"] = rgb_to_lab(rgb) if rgb is not None else None
            hold["spatial_evidence"] = None
            if hold.get("mask_path"):
                hold["spatial_evidence"] = estimate_polycam_spatial_evidence(
                    Path(frame["source_image"]),
                    frame_path.parent / hold["mask_path"],
                    frame.get("rotation_clockwise_degrees", 0),
                )
                spatial_observation_count += int(hold["spatial_evidence"] is not None)
            if hold["object_kind"] == "hold" and hold["color_lab"] is not None:
                observations.append(hold)

    clusters = _cluster_route_candidates(observations, color_distance_threshold)
    route_candidates = []
    for index, cluster in enumerate(clusters, 1):
        representative_rgb = np.median(
            [member["median_rgb"] for member in cluster.members], axis=0
        ).round().astype(int).tolist()
        route_candidates.append({
            "route_candidate_id": f"route_candidate_{index:04d}",
            "representative_rgb": representative_rgb,
            "representative_lab": cluster.center.round(3).tolist(),
            "observation_count": len(cluster.members),
            "assignment_method": "color_lab_delta_e76",
            "color_evidence": sorted({
                member.get("color_method", "unknown") for member in cluster.members
            }),
            "needs_review": True,
        })

    scene_frames = []
    scene_observations = []
    for frame_path, frame in frames:
        frame_path.write_text(json.dumps(frame, indent=2, allow_nan=False) + "\n")
        _render_routes(run, frame_path, frame)
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

    scene = {
        "schema_version": "2.0-draft",
        "wall_capture_id": run.name,
        "coordinate_space": "image",
        "route_candidate_assumption": "one_color_group_proposes_one_route_with_human_review",
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
            "glb_registration": "not_registered",
            "fusion_distance_m": fusion_distance_m,
        },
    }
    (run / "scene.json").write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n")
    return scene
