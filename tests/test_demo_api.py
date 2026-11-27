"""HTTP demo contract exercised with a controlled inference worker."""

import json
from pathlib import Path
from threading import Event
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from hold_detection import api


def test_uploads_queue_and_return_mobile_artifacts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    root = tmp_path / "runs"
    monkeypatch.setenv("DEMO_SAM_WEIGHTS", str(checkpoint))
    monkeypatch.setenv("DEMO_API_ROOT", str(root))
    monkeypatch.setenv("DEMO_API_TOKEN", "demo-fixture")
    monkeypatch.setenv("DEMO_MAX_PENDING_JOBS", "2")
    first_started = Event()
    release_first = Event()

    def inference(command, **kwargs):
        mode = command[command.index("--mode") + 1]
        result = Path(command[command.index("--output") + 1])
        result.mkdir()
        assert Path(command[4]).read_bytes() == b"fixture"
        wall = {"glb_path": None}
        if mode == "2d":
            first_started.set()
            assert release_first.wait(5)
            frame = result / "inference" / "frame"
            frame.mkdir(parents=True)
            (frame / "holds.json").write_text('{"processed_size_wh":[320,480]}')
        else:
            glb = result / "wall_upright.glb"
            glb.write_bytes(b"glTF fixture")
            wall["glb_path"] = str(glb)
        (result / "scene.json").write_text(json.dumps({
            "schema_version": "5.0", "analysis_mode": mode, "wall": wall,
            "observations": [{"source_image": str(result / "private.jpg"), "bbox_xyxy": [0, 0, 16, 16]}],
        }))

    monkeypatch.setattr(api.subprocess, "run", inference)
    path = api.PREFIX
    with TestClient(api.create_app(), headers={"Authorization": "Bearer demo-fixture"}) as client:
        try:
            assert client.post(path, headers={"Authorization": "wrong"}).status_code == 401
            wrong = client.post(path, data={"mode": "2d"}, files={"file": ("wall.mov", b"fixture")})
            assert wrong.status_code == 415
            photo = client.post(path, data={"mode": "2d"}, files={"file": ("wall.jpg", b"fixture")})
            assert photo.status_code == 202
            assert first_started.wait(5)
            video = client.post(path, data={"mode": "3d"}, files={"file": ("wall.mov", b"fixture")})
            assert video.status_code == 202
            photo_url = photo.json()["status_url"]
            video_url = video.json()["status_url"]
            assert client.get(photo_url).json()["status"] == "running"
            assert client.get(video_url).json()["status"] == "queued"
            assert client.get(video_url + "/wall.glb").status_code == 404
            crowded = client.post(path, data={"mode": "2d"}, files={"file": ("extra.jpg", b"fixture")})
            assert crowded.status_code == 429
        finally:
            release_first.set()

        complete = []
        for url in (photo_url, video_url):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                data = client.get(url).json()
                if data["status"] == "succeeded":
                    complete.append(data)
                    break
                assert data["status"] in {"queued", "running"}
                time.sleep(.01)
            else:
                pytest.fail(f"Analysis {url} did not complete")

        photo_data, video_data = complete
        assert photo_data["image"] == {"width": 320, "height": 480}
        assert photo_data["glb_url"] is None
        assert "source_image" not in photo_data["scene"]["observations"][0]
        assert video_data["scene"]["wall"]["glb_path"] == video_data["glb_url"]
        downloaded = client.get(video_data["glb_url"])
        assert downloaded.headers["content-type"] == "model/gltf-binary"
        assert downloaded.content == b"glTF fixture"

    with TestClient(api.create_app(), headers={"Authorization": "Bearer demo-fixture"}) as restarted:
        assert restarted.get(video_url).json()["status"] == "succeeded"
