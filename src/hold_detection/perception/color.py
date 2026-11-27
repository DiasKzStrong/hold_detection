"""Assign observed hold colors and group them in CIE Lab."""

from dataclasses import dataclass, field

import numpy as np

from ..capture.images import processed_image
from .geometry import crop_box


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


def measure_frame_colors(frames):
    """Attach observed RGB and Lab colors to the entries in each frame."""
    for frame in frames:
        processed = None
        for hold in frame["holds"]:
            rgb = hold.get("median_rgb")
            if rgb is None:
                if processed is None:
                    processed = processed_image(frame)
                bounds = crop_box(hold["bbox_xyxy"], *processed.size, padding=0)
                left, top, right, bottom = bounds
                crop = np.asarray(processed.crop(bounds)) if right > left and bottom > top else np.empty(0)
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


def assign_color_groups(frames, color_distance_threshold=22.0):
    """Return color-based route candidates for legacy comparisons."""
    measure_frame_colors(frames)
    observations = [hold for frame in frames for hold in frame["holds"]
                    if hold["object_kind"] == "hold" and hold["color_lab"] is not None]
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
    return route_candidates
