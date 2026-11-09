"""COLMAP reconstruction, calibrated depth lifting and explicit metric scale."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def read_colmap_depth(path):
    """Decode COLMAP's width&height&channels& float32 depth-map format."""
    with Path(path).open("rb") as stream:
        fields = []
        while len(fields) < 3:
            field = bytearray()
            while True:
                character = stream.read(1)
                if not character:
                    raise ValueError("Truncated COLMAP depth header")
                if character == b"&":
                    break
                field.extend(character)
                if len(field) > 20:
                    raise ValueError("Invalid COLMAP depth header")
            fields.append(int(field))
        width, height, channels = fields
        values = np.fromfile(stream, dtype="<f4")
    if channels != 1 or width < 1 or height < 1 or len(values) != width * height:
        raise ValueError("Expected a complete single-channel COLMAP depth map")
    return values.reshape((width, height), order="F").T


def read_colmap_cameras(folder):
    """Read undistorted intrinsics and world-to-camera poses, including blank track lines."""
    folder = Path(folder)
    cameras = {}
    for line in (folder / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        camera_id, model, width, height, *parameters = line.split()
        p = list(map(float, parameters))
        if model == "PINHOLE":
            fx, fy, cx, cy = p
        elif model == "SIMPLE_PINHOLE":
            fx, cx, cy = p
            fy = fx
        else:
            raise ValueError(f"Expected undistorted PINHOLE cameras, received {model}")
        cameras[int(camera_id)] = {"width": int(width), "height": int(height),
                                   "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    images, tracks = {}, False
    for line in (folder / "images.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        if tracks:
            tracks = False
            continue
        if not line.strip():
            continue
        tokens = line.split(maxsplit=9)
        if len(tokens) != 10:
            raise ValueError("Malformed COLMAP image pose")
        _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = tokens
        rotation = Rotation.from_quat(list(map(float, [qx, qy, qz, qw]))).as_matrix()
        images[name] = {**cameras[int(camera_id)], "rotation": rotation.tolist(),
                        "translation": list(map(float, [tx, ty, tz]))}
        tracks = True
    return images


def reconstruct_wall(capture, output, colmap="colmap", quality="medium", scale_m_per_unit=None, scale_reference=None):
    """Run resumable COLMAP SfM/MVS and export its mesh with trimesh."""
    import trimesh
    capture, output = Path(capture).resolve(), Path(output).resolve()
    manifest = json.loads((capture / "capture.json").read_text())
    if manifest["settings"]["mode"] != "3d":
        raise ValueError("Reconstruction requires a 3D capture")
    if quality not in {"low", "medium", "high", "extreme"}:
        raise ValueError("Unknown COLMAP quality")
    if scale_m_per_unit is not None and (not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0):
        raise ValueError("Metric scale must be finite and positive")
    reference = json.loads(Path(scale_reference).read_text()) if scale_reference else None
    if reference is not None and scale_m_per_unit is not None:
        raise ValueError("Supply a known-length reference or an explicit scale, not both")
    settings = {"capture": str(capture), "capture_identity": manifest["settings"]["identity"],
                "quality": quality, "scale_m_per_unit": scale_m_per_unit, "scale_reference": reference}
    output.mkdir(parents=True, exist_ok=True)
    spec = output / "settings.json"
    if spec.exists() and json.loads(spec.read_text()) != settings:
        raise ValueError("Reconstruction settings changed; choose another output")
    if not spec.exists() and any(output.iterdir()):
        raise FileExistsError("Reconstruction output contains unrelated files")
    spec.write_text(json.dumps(settings, indent=2) + "\n")
    completed = output / "reconstruction.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if all(Path(result[key]).exists() for key in ["mesh_path", "glb_path", "image_source", "camera_model"]):
            return result
        raise ValueError("Completed reconstruction has missing artifacts")
    executable = shutil.which(str(colmap))
    if not executable:
        raise RuntimeError("COLMAP executable not found. Load a CUDA COLMAP module/container and set --colmap.")

    def command(arguments, name):
        with (output / f"{name}.log").open("a") as log:
            result = subprocess.run([executable, *map(str, arguments)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"COLMAP {name} failed ({result.returncode}); inspect {output / (name + '.log')}")

    command(["automatic_reconstructor", "--workspace_path", output,
             "--image_path", manifest["image_source"], "--data_type",
             "video" if manifest["input_kind"] == "video" else "individual",
             "--quality", quality, "--use_gpu", "1", "--dense", "1", "--mesher", "poisson"], "reconstruct")
    # Keep one connected reconstruction; never mix coordinates from disconnected SfM components.
    models = sorted(p for p in (output / "dense").glob("*") if (p / "meshed-poisson.ply").is_file())
    if not models:
        raise RuntimeError("COLMAP produced no dense mesh; inspect overlap, sharpness and reconstruct.log")
    dense = max(models, key=lambda p: (p / "fused.ply").stat().st_size)
    text = dense / "cameras_text"
    text.mkdir(exist_ok=True)
    command(["model_converter", "--input_path", dense / "sparse",
             "--output_path", text, "--output_type", "TXT"], "export_cameras")
    cameras = read_colmap_cameras(text)
    if len(cameras) < 3:
        raise RuntimeError("Fewer than three images registered in the selected COLMAP component")
    mesh = trimesh.load(dense / "meshed-poisson.ply", force="mesh", process=False)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise RuntimeError("Reconstructed mesh is empty")
    if reference is not None:
        name = reference["image"]
        depth = read_colmap_depth(dense / "stereo/depth_maps" / f"{name}.geometric.bin")
        scale_m_per_unit = scale_from_reference(depth, cameras[name], reference)
    if scale_m_per_unit is not None:
        mesh.apply_scale(scale_m_per_unit)
    glb = output / "wall.glb"
    mesh.export(glb)
    result = {"schema_version": "1.0", "method": "colmap_sfm_mvs_poisson",
              "image_source": str(dense / "images"), "camera_model": str(text),
              "depth_directory": str(dense / "stereo/depth_maps"),
              "mesh_path": str(dense / "meshed-poisson.ply"), "glb_path": str(glb),
              "registered_images": len(cameras), "capture_images": len(manifest["frames"]),
              "registered_image_names": sorted(cameras), "component_count": len(models),
              "scale_m_per_unit": scale_m_per_unit,
              "scale_reference": reference,
              "coordinate_frame": "colmap_world", "up_axis": None,
              "units": "meters" if scale_m_per_unit is not None else "arbitrary",
              "limitations": ["single_connected_component", "gravity_not_estimated", "occluded_grip_geometry_unknown"]}
    completed.write_text(json.dumps(result, indent=2) + "\n")
    return result


def reference_camera_points(depth, camera, pixels, require_support=False):
    """Unproject two reference pixels in the undistorted camera frame."""
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape != (2, 2) or not np.isfinite(pixels).all():
        raise ValueError("Expected two finite reference pixels")
    positions = []
    for x, y in pixels:
        if not 0 <= x < camera["width"] or not 0 <= y < camera["height"]:
            raise ValueError("Reference pixels lie outside the undistorted image")
        row, column = int(y * depth.shape[0] / camera["height"]), int(x * depth.shape[1] / camera["width"])
        z = float(depth[row, column])
        if not np.isfinite(z) or z <= 0:
            raise ValueError("Reference has no valid stereo depth at a clicked point")
        if require_support:
            patch = depth[max(0, row - 1):row + 2, max(0, column - 1):column + 2]
            agreeing = np.isfinite(patch) & (np.abs(patch - z) <= 0.05 * z)
            if np.count_nonzero(agreeing) < max(3, patch.size // 2 + 1):
                raise ValueError("Reference depth is inconsistent near a clicked point; choose another point or frame")
        positions.append([(x - camera["cx"]) * z / camera["fx"],
                          (y - camera["cy"]) * z / camera["fy"], z])
    return np.asarray(positions)


def scale_from_reference(depth, camera, reference):
    """Calibrate from two undistorted-image pixels on a visible known-length reference."""
    first, second = reference_camera_points(
        depth, camera, [reference["point_a_px"], reference["point_b_px"]],
    )
    length = float(np.linalg.norm(second - first))
    distance_m = float(reference["distance_m"])
    if length <= 1e-9 or not np.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("Reference length must be finite and positive with distinct endpoints")
    return distance_m / length


def lift_mask(depth, mask, camera, maximum_samples=128):
    """Lift a mask in undistorted image coordinates through visible multiview stereo depth."""
    depth = np.asarray(depth, dtype=float)
    resized = np.asarray(Image.fromarray(np.uint8(mask) * 255).resize(
        (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)) > 0
    valid_depth = np.isfinite(depth) & (depth > 0)
    selected = resized & valid_depth
    if np.count_nonzero(selected) < 3:
        return None

    def points(selection):
        y, x = np.nonzero(selection)
        z = depth[selection]
        pixels_x = (x + 0.5) * camera["width"] / depth.shape[1]
        pixels_y = (y + 0.5) * camera["height"] / depth.shape[0]
        cam = np.column_stack([(pixels_x - camera["cx"]) * z / camera["fx"],
                               (pixels_y - camera["cy"]) * z / camera["fy"], z])
        # COLMAP X_camera = R * X_world + t; row-vector inverse is (X_camera - t) * R.
        return (cam - np.asarray(camera["translation"])) @ np.asarray(camera["rotation"])

    cloud = points(selected)
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(resized, iterations=4) & ~resized & valid_depth
    normal, protrusion = None, None
    if np.count_nonzero(ring) >= 6:
        wall = points(ring)
        center = np.median(wall, axis=0)
        _, _, axes = np.linalg.svd(wall - center, full_matrices=False)
        vector = axes[-1]
        camera_center = -np.asarray(camera["translation"]) @ np.asarray(camera["rotation"])
        if np.dot(vector, camera_center - center) < 0:
            vector = -vector
        normal = vector.tolist()
        protrusion = max(0.0, float(np.quantile((cloud - center) @ vector, 0.95)))
    indices = np.linspace(0, len(cloud) - 1, min(len(cloud), maximum_samples), dtype=int)
    return {"position": np.median(cloud, axis=0).tolist(),
            "extent": np.ptp(cloud, axis=0).tolist(), "samples": cloud[indices].tolist(),
            "wall_normal": normal, "protrusion": protrusion,
            "depth_valid_fraction": float(np.count_nonzero(selected) / max(1, np.count_nonzero(resized))),
            "method": "colmap_masked_depth_unprojection"}
