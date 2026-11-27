"""Serve new wall analyses through a single GPU worker."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
from secrets import compare_digest
import shutil
import subprocess
import sys
from threading import BoundedSemaphore
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .capture.images import IMAGE_EXTENSIONS
from .capture.prepare import VIDEO_EXTENSIONS
from .scene.artifacts import SCENE_SCHEMA_VERSION, read_scene, read_single_image_size


logger = logging.getLogger(__name__)
PREFIX = "/api/v1/analyses"
PRIVATE_FIELDS = {
    "source_image", "mask_path", "image_source", "camera_model", "depth_directory",
    "mesh_path", "glb_path",
}


def write_status(job, mode, status, **fields):
    pending = job / "status.tmp"
    pending.write_text(json.dumps({"id": job.name, "mode": mode, "status": status, **fields}))
    pending.replace(job / "status.json")


def public_scene(value):
    """Keep scene measurements and omit references to server files."""
    if isinstance(value, dict):
        return {key: public_scene(item) for key, item in value.items() if key not in PRIVATE_FIELDS}
    if isinstance(value, list):
        return [public_scene(item) for item in value]
    return value


def wall_file(result, scene):
    declared = scene["wall"].get("glb_path")
    if declared:
        candidate = Path(declared).resolve()
        if candidate.is_relative_to(result.resolve()) and candidate.suffix.lower() == ".glb" and candidate.is_file():
            return candidate
    raise ValueError("Missing or invalid GLB artifact")


def run_analysis(job, media, mode, options, worker_env, slots):
    try:
        write_status(job, mode, "running")
        result = job / "result"
        command = [sys.executable, "-m", "hold_detection.cli", "analyze-wall", str(media),
                   "--mode", mode, "--output", str(result), *options]
        with (job / "worker.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env=worker_env)
        scene = read_scene(result)
        if scene.get("analysis_mode") != mode or scene.get("schema_version") != SCENE_SCHEMA_VERSION:
            raise ValueError("Unexpected scene format")

        glb_url = None
        image = None
        if mode == "3d":
            wall_file(result, scene)
            glb_url = f"{PREFIX}/{job.name}/wall.glb"
        else:
            width, height = read_single_image_size(result)
            image = {"width": width, "height": height}

        exported = public_scene(scene)
        exported["wall"]["glb_path"] = glb_url
        (job / "response.json").write_text(json.dumps({
            "scene": exported, "glb_url": glb_url, "image": image,
        }, allow_nan=False))
        write_status(job, mode, "succeeded")
    except Exception:
        logger.exception("Wall analysis %s failed. Details are in %s", job.name, job / "worker.log")
        write_status(job, mode, "failed", error="Wall analysis failed. Contact the demo operator with the job ID.")
    finally:
        slots.release()


def create_app():
    root = Path(os.environ.get("DEMO_API_ROOT", "runs/api")).resolve()
    token = os.environ.get("DEMO_API_TOKEN")
    sam = os.environ.get("DEMO_SAM_WEIGHTS")
    if not token:
        raise RuntimeError("Set DEMO_API_TOKEN before starting the API")
    if not sam or not Path(sam).is_file():
        raise RuntimeError("Set DEMO_SAM_WEIGHTS to an existing SAM 3 checkpoint")

    device = os.environ.get("DEMO_GPU_DEVICE", "0")
    if not device.isascii() or not device.isdecimal():
        raise RuntimeError("DEMO_GPU_DEVICE must be a nonnegative GPU index")
    index = int(device)
    worker_env = os.environ.copy()
    if visible := worker_env.get("CUDA_VISIBLE_DEVICES"):
        devices = visible.split(",")
        if index >= len(devices) or devices[index].strip() in {"", "-1"}:
            raise RuntimeError("DEMO_GPU_DEVICE is outside the visible CUDA devices")
        device = devices[index].strip()
    elif visible == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides all GPUs")
    worker_env["CUDA_VISIBLE_DEVICES"] = device
    options = ["--sam-weights", str(Path(sam).resolve()), "--device", "0",
               "--colmap", os.environ.get("DEMO_COLMAP", "colmap")]
    if type_weights := os.environ.get("DEMO_TYPE_WEIGHTS"):
        if not Path(type_weights).is_file():
            raise RuntimeError("DEMO_TYPE_WEIGHTS must point to an existing checkpoint")
        options += ["--type-weights", str(Path(type_weights).resolve()), "--type-imgsz",
                    os.environ.get("DEMO_TYPE_IMGSZ", "640")]

    max_upload = int(os.environ.get("DEMO_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    max_jobs = int(os.environ.get("DEMO_MAX_PENDING_JOBS", "8"))
    if max_upload <= 0 or max_jobs <= 0:
        raise RuntimeError("Upload size and pending-job limits must be positive")
    slots = BoundedSemaphore(max_jobs)

    @asynccontextmanager
    async def lifespan(app):
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".server.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("An API server already uses this run directory") from error
            for path in root.glob("*/status.json"):
                saved = json.loads(path.read_text())
                if saved["status"] in {"queued", "running"}:
                    write_status(path.parent, saved["mode"], "failed", error="The API restarted during analysis.")
            with ThreadPoolExecutor(max_workers=1) as worker:
                app.state.worker = worker
                yield

    app = FastAPI(title="DeMoSearch Demo API", lifespan=lifespan)

    @app.middleware("http")
    async def protect_analyses(request: Request, call_next):
        if request.url.path.startswith(PREFIX):
            supplied = request.headers.get("Authorization", "").encode()
            if not compare_digest(supplied, f"Bearer {token}".encode()):
                return JSONResponse({"detail": "A valid demo token is required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            if request.method == "POST":
                length = request.headers.get("Content-Length")
                if length is None or not length.isdecimal():
                    return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
                if int(length) > max_upload + 1024 * 1024:
                    return JSONResponse({"detail": "Upload exceeds the configured limit"}, status_code=413)
        return await call_next(request)

    @app.post(PREFIX, status_code=202)
    def submit(mode: Annotated[Literal["2d", "3d"], Form()], file: Annotated[UploadFile, File()]):
        extension = Path(file.filename or "").suffix.lower()
        allowed = IMAGE_EXTENSIONS if mode == "2d" else VIDEO_EXTENSIONS
        if extension not in allowed:
            raise HTTPException(415, "Upload a photo for 2D or a video for 3D")
        if not file.size:
            raise HTTPException(400, "The upload is empty")
        if file.size > max_upload:
            raise HTTPException(413, "Upload exceeds the configured limit")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "The demo queue is full", headers={"Retry-After": "5"})

        job = root / str(uuid4())
        try:
            job.mkdir()
            media = job / f"input{extension}"
            with media.open("xb") as stream:
                shutil.copyfileobj(file.file, stream)
            write_status(job, mode, "queued")
            app.state.worker.submit(run_analysis, job, media, mode, options, worker_env, slots)
        except Exception:
            slots.release()
            if job.is_dir():
                shutil.rmtree(job)
            raise
        return {"id": job.name, "mode": mode, "status": "queued", "status_url": f"{PREFIX}/{job.name}"}

    def job_status(job_id):
        job = root / str(job_id)
        path = job / "status.json"
        if not path.is_file():
            raise HTTPException(404, "Unknown analysis")
        return job, json.loads(path.read_text())

    @app.get(f"{PREFIX}/{{job_id}}")
    def following(job_id: UUID):
        job, status = job_status(job_id)
        if status["status"] == "succeeded":
            status.update(json.loads((job / "response.json").read_text()))
        return status

    @app.get(f"{PREFIX}/{{job_id}}/wall.glb")
    def download_wall(job_id: UUID):
        job, status = job_status(job_id)
        if status["mode"] != "3d" or status["status"] != "succeeded":
            raise HTTPException(404, "A completed 3D wall is required")
        result = job / "result"
        scene = read_scene(result)
        try:
            path = wall_file(result, scene)
        except ValueError as error:
            raise HTTPException(404, "Wall artifact is unavailable") from error
        return FileResponse(path, media_type="model/gltf-binary", filename="wall.glb")

    return app
