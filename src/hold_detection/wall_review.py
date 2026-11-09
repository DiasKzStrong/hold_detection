"""Image overlays and mesh correspondences for a saved wall scene."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import COLORS
from .scene import ROUTE_COLORS
from .visualization import processed_image


GEOMETRY_COLORS = {"multi_view": "#34d399", "single_view": "#fbbf24", "unmapped": "#94a3b8"}
UNKNOWN_COLOR = "#60a5fa"


def hold_labels(scene):
    """Use the same short display label for a hold on photos and on the mesh."""
    return {hold["id"]: f"H{index:03d}" for index, hold in enumerate(scene["holds"], 1)}


def render_frame_overlay(run, frame, scene, labels, color_by="type", selected_hold=None):
    """Show the original processed photo and its actual saved instance masks."""
    original = processed_image(frame)
    width, height = original.size
    original.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    ratio_x, ratio_y = original.width / width, original.height / height
    overlay = original.copy()
    by_observation = {obs_id: hold for hold in scene["holds"] for obs_id in hold["observation_ids"]}
    groups = sorted({hold["color_group_id"] for hold in scene["holds"] if hold["color_group_id"]})
    group_colors = {group: ROUTE_COLORS[i % len(ROUTE_COLORS)] for i, group in enumerate(groups)}
    annotations = []
    for observation in scene["observations"]:
        if observation["frame_id"] != frame["frame_id"]:
            continue
        hold = by_observation[observation["id"]]
        if selected_hold is not None and hold["id"] != selected_hold:
            continue
        color = COLORS.get(hold["hold_type"], UNKNOWN_COLOR)
        if color_by == "group":
            color = group_colors.get(hold["color_group_id"], UNKNOWN_COLOR)
        if color_by == "geometry":
            color = GEOMETRY_COLORS[hold["geometry_status"]]
        if observation["mask_path"]:
            with Image.open(Path(run) / observation["mask_path"]) as stored:
                mask = stored.convert("L").resize(original.size, Image.Resampling.NEAREST)
            alpha = mask.point(lambda value: 120 if value else 0)
            overlay = Image.composite(Image.new("RGB", original.size, color), overlay, alpha)
        x1, y1, x2, y2 = observation["bbox_xyxy"]
        box = (x1 * ratio_x, y1 * ratio_y, x2 * ratio_x, y2 * ratio_y)
        caption = labels[hold["id"]]
        if selected_hold is not None:
            caption += " " + hold["hold_type"]
        annotations.append((box, color, caption))

    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=13 if selected_hold is not None else 10)
    for box, color, caption in annotations:
        draw.rectangle(box, outline=color, width=2)
        anchor = (box[0], max(0, box[1] - 13))
        draw.rectangle(draw.textbbox(anchor, caption, font=font), fill="#131a24")
        draw.text(anchor, caption, font=font, fill=color)
    return original, overlay, len(annotations)


def wall_mesh_figure(scene, labels, face_budget=120_000):
    """Render the exported GLB and the same hold IDs in its coordinate frame."""
    import plotly.graph_objects as go
    import trimesh

    mesh = trimesh.load_mesh(scene["wall"]["glb_path"], process=False)
    if len(mesh.faces) > face_budget:
        mesh = mesh.simplify_quadric_decimation(face_count=face_budget)
    x, y, z = np.asarray(mesh.vertices, dtype=np.float32).T
    i, j, k = np.asarray(mesh.faces, dtype=np.int32).T
    figure = go.Figure(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color="#73859a",
                                 opacity=0.75, name="Wall surface", hoverinfo="skip"))
    scale = scene["wall"]["scale_m_per_unit"] or 1.0
    links = [[], [], []]
    for status, name in [("multi_view", "Matched across views"), ("single_view", "Single view")]:
        points, captions = [], []
        for hold in scene["holds"]:
            if hold["geometry_status"] != status or hold["position_3d"] is None:
                continue
            center = np.asarray(hold["position_3d"]) * scale
            correspondence = hold["mesh_correspondence"]
            anchor = np.asarray(correspondence["position_3d"]) * scale if correspondence else center
            distance = ""
            if correspondence:
                gap_m = correspondence["distance_m"]
                gap = (f"{gap_m:.3g} m" if gap_m is not None else
                       f"{correspondence['distance_reconstruction_units']:.3g} reconstruction units")
                distance = f"<br>Center to mesh vertex: {gap}"
                for axis, values in enumerate(links):
                    values.extend([center[axis], anchor[axis], None])
            points.append(anchor)
            height = hold.get("height_3d")
            height_text = f"<br>Relative height: {height * scale:.3g} {scene['wall']['units']}" if height is not None else ""
            captions.append(f"{labels[hold['id']]} · {hold['hold_type']}"
                            f"<br>{hold['observation_count']} view(s){height_text}{distance}")
        if points:
            px, py, pz = np.asarray(points).T
            figure.add_scatter3d(x=px, y=py, z=pz, mode="markers", name=name,
                                 marker=dict(size=5, color=GEOMETRY_COLORS[status]),
                                 hovertext=captions, hoverinfo="text")
    if links[0]:
        figure.add_scatter3d(x=links[0], y=links[1], z=links[2], mode="lines",
                             name="Lifted centers to mesh vertices", visible="legendonly",
                             line=dict(color="#f8fafc", width=3), hoverinfo="skip")
    figure.update_layout(
        template="plotly_dark", height=650, margin=dict(l=0, r=0, t=35, b=0),
        title=f"Wall GLB · {scene['wall']['units']}",
        scene=dict(aspectmode="data"),
        legend=dict(orientation="h", x=0, y=0.02),
    )
    up = scene["wall"].get("up_vector")
    if up is not None:
        figure.update_scenes(camera_up=dict(zip("xyz", up)))
    return figure
