import ast
import base64
import hashlib
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
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


def test_notebook_is_valid_unexecuted_python_with_current_embedded_sources():
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
    install = next(c.source for c in notebook.cells if c.id == "install")
    assert "system_site_packages=True" in install
    assert "torch==" not in install  # Do not replace Colab's paired CUDA wheels.


def test_embedded_bootstrap_extracts_exact_source_without_cloning_project(tmp_path):
    notebook = load_notebook()
    source = next(c.source for c in notebook.cells if c.id == "bootstrap")

    class LocalWorkPath(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == "/content/demosearch":
                return ast.copy_location(ast.Constant(str(tmp_path / "work")), node)
            return node

    tree = ast.fix_missing_locations(LocalWorkPath().visit(ast.parse(source)))
    namespace = {"sys": sys}
    original_path = list(sys.path)
    try:
        exec(compile(tree, "bootstrap", "exec"), namespace)
        for relative, digest in notebook.metadata.demosearch.embedded_files.items():
            content = (namespace["PROJECT"] / relative).read_bytes()
            assert hashlib.sha256(content).hexdigest() == digest
            assert content == (ROOT / relative).read_bytes()
        # A rerun is harmless, but a local source edit must not be overwritten.
        exec(compile(tree, "bootstrap", "exec"), namespace)
        edited = namespace["PROJECT"] / "src/hold_detection/__init__.py"
        edited.write_text("# user edit\n")
        with pytest.raises(FileExistsError):
            exec(compile(tree, "bootstrap", "exec"), namespace)
        assert edited.read_text() == "# user edit\n"
    finally:
        sys.path[:] = original_path


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
