import ast
from datetime import datetime, timezone
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
from zipfile import ZipFile

import nbformat
from PIL import Image
import pytest

from hold_detection.colab_support import bundle_results, prepare_inputs
from hold_detection.geometry import Detection
from hold_detection.pipeline import run_inference

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DeMoSearch_SAM3_Colab.ipynb"


def load_notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_is_valid_readable_and_unexecuted():
    notebook = load_notebook()
    nbformat.validate(notebook)
    ids = [c.id for c in notebook.cells]
    assert len(ids) == len(set(ids))
    for c in notebook.cells:
        if c.cell_type == "code":
            ast.parse(c.source, filename=c.id)
            assert c.outputs == [] and c.execution_count is None
    spec = importlib.util.spec_from_file_location("notebook_builder", ROOT / "scripts/build_colab_notebook.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert json.loads(NOTEBOOK.read_text()) == builder.build_notebook()
    sources = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "PAYLOAD_B64" not in sources
    assert ids.index("baseline") < ids.index("sam-setup")
    assert {"viewer", "crop-grid", "metrics", "gt-upload", "export"} <= set(ids)
    assert notebook.metadata.demosearch.experiment_version == 2


def test_experiment_settings_execute_without_gpu_or_authentication(tmp_path):
    notebook = load_notebook()
    source = next(c.source for c in notebook.cells if c.id == "settings")
    namespace = {"prepared": SimpleNamespace(kind="polycam"), "PROJECT": tmp_path,
                 "datetime": datetime, "timezone": timezone, "uuid": uuid, "Path": Path}
    exec(compile(source, "settings", "exec"), namespace)
    assert namespace["rotation"] == 90
    assert namespace["RUNS"] == {}
    assert namespace["EXPERIMENT"].is_dir()
    assert namespace["CANDIDATE_FLOOR"] < namespace["YOLO_THRESHOLD"]


def test_bootstrap_and_sam_setup_never_import_kernel_torch():
    notebook = load_notebook()
    for cell_id in ("bootstrap", "sam-setup"):
        source = next(c.source for c in notebook.cells if c.id == cell_id)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "torch" for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch"
        assert "torch." not in source
    bootstrap = next(c.source for c in notebook.cells if c.id == "bootstrap")
    assert "MODEL_ENV = runtime_info(PYTHON, PROJECT)" in bootstrap
    assert "Select a Python 3.12 runtime" not in bootstrap


@pytest.mark.parametrize("error_message,is_oom", [
    ("TypeError: Got unsupported ScalarType BFloat16", False),
    ("CUDA out of memory. Tried to allocate a tensor", True),
])
def test_sam_failure_suggests_larger_gpu_only_for_cuda_oom(tmp_path, capsys, error_message, is_oom):
    def fail(*args, **kwargs):
        raise RuntimeError(error_message)
    source = next(c.source for c in load_notebook().cells if c.id == "sam3-run")
    namespace = {"RUNS": {}, "THRESHOLDS": {}, "SAM_READY": True, "EXPERIMENT": tmp_path,
                 "uuid": uuid, "SOURCE": tmp_path, "rotation": 0,
                 "HOLD_PROMPTS": "climbing hold, climbing foothold",
                 "VOLUME_PROMPTS": "climbing volume",
                 "CANDIDATE_FLOOR": 0.05, "MAX_IMAGES": 1, "run_holds": fail}
    exec(compile(source, "sam3-run", "exec"), namespace)
    output = capsys.readouterr().out
    assert error_message in output
    assert ("try a larger GPU" in output) == is_oom
    assert namespace["RUNS"] == {}


def jpeg_bytes():
    stream = BytesIO()
    Image.new("RGB", (20, 15), (30, 60, 90)).save(stream, format="JPEG")
    return stream.getvalue()


def test_polycam_import_never_uses_depth_pngs_as_photos(tmp_path):
    archive = tmp_path / "raw.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("keyframes/corrected_images/123.jpg", jpeg_bytes())
        z.writestr("keyframes/images/123.jpg", jpeg_bytes())
        for folder in ("cameras", "corrected_cameras"):
            z.writestr(f"keyframes/{folder}/123.json", "{}")
        for folder in ("depth", "confidence"):
            z.writestr(f"keyframes/{folder}/123.png", b"not RGB")
    prepared = prepare_inputs([archive], tmp_path / "prepared")
    assert prepared.kind == "polycam" and prepared.image_count == 1
    assert prepared.source.name == "corrected_images"
    assert list(prepared.source.iterdir())[0].read_bytes() == jpeg_bytes()
    assert prepare_inputs([tmp_path / "prepared"], tmp_path / "unused").source == prepared.source
    with pytest.raises(FileExistsError):
        prepare_inputs([archive], tmp_path / "prepared")


def test_partial_polycam_and_generic_photo_zip_are_supported(tmp_path):
    partial = tmp_path / "partial.zip"
    with ZipFile(partial, "w") as z:
        z.writestr("keyframes/corrected_images/1.jpg", jpeg_bytes())
        z.writestr("keyframes/depth/1.png", b"not RGB")
    prepared = prepare_inputs([partial], tmp_path / "partial")
    assert prepared.kind == "polycam" and prepared.image_count == 1
    photos = tmp_path / "photos.zip"
    with ZipFile(photos, "w") as z:
        z.writestr("photos/wall.jpg", jpeg_bytes())
        z.writestr("__MACOSX/._wall.jpg", b"not a photo")
        z.writestr("notes.txt", "notes")
    prepared = prepare_inputs([photos], tmp_path / "photos")
    assert prepared.kind == "photos" and prepared.image_count == 1


def test_uploads_preserve_originals_and_reject_mesh_and_unsafe_zips(tmp_path):
    photo = tmp_path / "wall.jpg"
    photo.write_bytes(jpeg_bytes())
    prepared = prepare_inputs([photo], tmp_path / "prepared")
    assert prepared.image_count == 1 and photo.read_bytes() == jpeg_bytes()
    assert list(prepared.source.iterdir())[0].read_bytes() == jpeg_bytes()
    for name, entries, expected in [
        ("mesh.zip", {"wall.obj": b"v 0 0 0", "texture.jpg": jpeg_bytes()}, "mesh/texture"),
        ("bad.zip", {"../escape.jpg": jpeg_bytes()}, "Unsafe"),
        ("depth.zip", {"keyframes/depth/1.png": b"depth"}, "no RGB"),
    ]:
        archive = tmp_path / name
        with ZipFile(archive, "w") as z:
            for path, data in entries.items():
                z.writestr(path, data)
        with pytest.raises(ValueError, match=expected):
            prepare_inputs([archive], tmp_path / (name + "_output"))
    with pytest.raises(ValueError, match="not both"):
        prepare_inputs([photo, tmp_path / "mesh.zip"], tmp_path / "mixed")


def test_result_zip_contains_offline_gallery_and_all_referenced_assets(tmp_path):
    image = tmp_path / "wall.jpg"
    image.write_bytes(jpeg_bytes())

    class Detector:
        def detect(self, image, **kwargs):
            return [Detection((2, 2, 10, 10), 0.8, "jug")]

    run = tmp_path / "run"
    run_inference(image, run, Detector())
    archive = bundle_results(run, tmp_path / "results.zip")
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert "run/index.html" in z.namelist()
        assert "run/summary.json" in z.namelist()
        assert any(n.endswith("/annotated.jpg") for n in z.namelist())
        assert any("/crops/" in n for n in z.namelist())
        assert not any("models/" in n or "token" in n.lower() for n in z.namelist())
    with pytest.raises(FileExistsError):
        bundle_results(run, archive)
    with pytest.raises(ValueError, match="outside"):
        bundle_results(run, run / "bundle.zip")


@pytest.mark.parametrize("with_gt", [False, True])
def test_actual_notebook_metrics_errors_and_export_cells(tmp_path, with_gt):
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from hold_detection.cli import main
    from hold_detection.visualization import ComparisonViewer, plot_comparison

    image = tmp_path / "wall.jpg"
    image.write_bytes(jpeg_bytes())

    class Detector:
        def detect(self, image, **kwargs):
            return [Detection((2, 2, 10, 10), 0.8, "jug")]

    experiment = tmp_path / "experiment"
    run = experiment / "yolo"
    run_inference(image, run, Detector(), model_metadata={"classifier": "detector"})
    runs = {"YOLO fixture": run}
    gt = None
    if with_gt:
        gt = tmp_path / "annotations.json"
        gt.write_text(json.dumps({
            "images": [{"id": 1, "file_name": "wall.jpg", "width": 20, "height": 15}],
            "categories": [{"id": 1, "name": "jug"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 8]}],
        }))

    def run_holds(*arguments, log_name):
        assert main([str(a) for a in arguments]) == 0

    shown, downloads = [], []
    ns = {"json": json, "uuid": uuid, "plt": plt, "HTML": HTML, "display": shown.append,
          "RUNS": runs, "EXPERIMENT": experiment, "GT_COORDINATES": "source", "GT_FILE": gt,
          "viewer": ComparisonViewer(runs, {"YOLO fixture": 0.5}), "run_holds": run_holds,
          "plot_comparison": plot_comparison, "CODE_SHA": "test_fixture", "RUN_SAM3": False,
          "SAM_READY": False, "runtime_info": lambda *_: {"gpu": None}, "PYTHON": None,
          "PROJECT": ROOT, "WORK": tmp_path, "files": SimpleNamespace(download=downloads.append)}
    notebook = load_notebook()
    for cell_id in ["metrics", "errors", "export"]:
        source = next(c.source for c in notebook.cells if c.id == cell_id)
        exec(compile(source, cell_id, "exec"), ns)
    assert ns["report"]["quality_status"] == ("evaluated" if with_gt else "ground_truth_required")
    assert ns["table"]["Candidates"].iloc[0] == 1
    assert "YOLO fixture" in shown[0].to_html()
    assert len(downloads) == 1
    with ZipFile(downloads[0]) as archive:
        assert archive.testzip() is None
        assert "experiment/session.json" in archive.namelist()
        assert any(n.endswith("/metrics.csv") for n in archive.namelist())
        assert any(n.endswith("/errors_000.jpg") for n in archive.namelist()) == with_gt
