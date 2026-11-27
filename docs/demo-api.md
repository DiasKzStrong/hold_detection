# Mobile demo HTTP API

The API accepts one new capture per request. An analysis job is a capture
submitted for processing with its own status, ID and result. One worker runs
the existing wall pipeline on the GPU. The API stores inputs, results and
worker logs under `DEMO_API_ROOT`, which defaults to `runs/api`.

## Start the server

Use the existing [CUDA wall pipeline environment](wall-pipeline.md#runtime) on
the GPU machine. Install the API dependencies with the same active Python.
Install CUDA COLMAP for 3D and an approved SAM 3 checkpoint for both modes.

```bash
python -m pip install -r scripts/requirements-api.txt
export DEMO_SAM_WEIGHTS=/path/to/sam3.pt
export DEMO_COLMAP=/path/to/colmap
export DEMO_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m uvicorn hold_detection.api:create_app --factory --host 0.0.0.0 --port 8000
```

For optional grip-type inference, set `DEMO_TYPE_WEIGHTS` to the YOLO checkpoint
and `DEMO_TYPE_IMGSZ` to its input size. The default is 640. A single GPU is
selected for SAM 3, YOLO and COLMAP with `DEMO_GPU_DEVICE`, which defaults to
`0`. The index is relative to the server process’s visible CUDA devices. Run one Uvicorn process
per results directory. Archive results before clearing that directory between demos.

On your local network, use `http://<computer-lan-ip>:8000` from the phone. For
an EC2 demo over the Internet, put the API behind an HTTPS endpoint and restrict
network access to the demo clients. Send `Authorization: Bearer <token>` on
every API request, including the GLB download.

## Upload, poll and download

```bash
curl -H "Authorization: Bearer $DEMO_API_TOKEN" \
  -F mode=2d -F file=@wall.jpg http://localhost:8000/api/v1/analyses
```

For 3D, use `mode=3d` and upload a video such as `wall.mov` or `wall.mp4`.
The multipart request must have a Content-Length. The upload limit defaults
to 200 MiB and the pending job limit to eight, including the running task.
Tune these with `DEMO_MAX_UPLOAD_MB` and `DEMO_MAX_PENDING_JOBS`. A full queue
returns HTTP 429.

1. A successful upload returns HTTP 202 with `id`, `mode`, `status` and `status_url`.
2. Poll `status_url` with GET. States are `queued`, `running`, `succeeded` and `failed`.
3. A successful poll includes the exported `scene` object, `image` dimensions
   for 2D and `glb_url` for 3D. Resolve relative URLs against the API base URL.
4. A failed job includes an `error`. Look up `<job-id>/worker.log` on the
   server for details. The server marks interrupted jobs as failed after restart.
   Resubmit the capture to try again. Completed results persist on disk.

In 2D, display the same EXIF-oriented photo you uploaded. If the user crops it
on the phone, upload and display that crop. Draw markers with normalized
`holds[].position` and rectangles with the pixel `observations[].bbox_xyxy`
scaled from the returned `image.width` and `image.height` to the displayed image.
Apply the displayed image's offset within its view when the view adds padding.
The server response includes the scene JSON and image dimensions.

In 3D, download `glb_url` to the phone with the same authorization header and
load the local file in the 3D viewer. Place markers at the non-null values in
`holds[].position_glb`. The GLB and positions use the same axes. Check `units`
before displaying distances. Uncalibrated runs use arbitrary units.
