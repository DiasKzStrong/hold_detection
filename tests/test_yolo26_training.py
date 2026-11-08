"""Exercise the notebook's dataset hand-off with real files, without a GPU."""

import ast
import json
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

import nbformat
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]


def training_module():
    from hold_detection import yolo_training
    return yolo_training


def export_fixture(root):
    root.mkdir()
    (root / "data.yaml").write_text(
        "names: [Jug, Crimp, Pinch, Pocket, Slope, Volume]\nnc: 6\n"
        "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
    )
    for index, split in enumerate(["train", "valid", "test"]):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        Image.new("RGB", (32, 24), (20 + index * 40, 10, 90)).save(root / split / "images/wall.png")
        (root / split / "labels/wall.txt").write_text("0 0.5 0.5 0.25 0.5\n")
    return root


def test_prepare_yolo_export_keeps_annotation_class_ids_and_source_splits(tmp_path):
    module = training_module()
    source = export_fixture(tmp_path / "export")
    result = module.prepare_yolo_dataset(source, tmp_path / "prepared")
    assert result["names"] == ["jug", "crimp", "pinch", "pocket", "sloper", "volume"]
    assert result["splits"]["val"] == {"images": 1, "boxes": 1, "class_counts": [1, 0, 0, 0, 0, 0]}
    assert (tmp_path / "prepared/val/labels/wall.txt").read_text().startswith("0 ")
    assert Path(result["data_yaml"]).is_file()
    assert result["dataset_sha256"]
    with pytest.raises(FileExistsError):
        module.prepare_yolo_dataset(source, tmp_path / "prepared")


@pytest.mark.parametrize("problem", ["missing_label", "invalid_box", "split_overlap", "missing_class"])
def test_training_rejects_bad_or_leaking_labels_before_gpu_allocation(tmp_path, problem):
    module = training_module()
    source = export_fixture(tmp_path / "export")
    if problem == "missing_label":
        (source / "train/labels/wall.txt").unlink()
    elif problem == "invalid_box":
        (source / "train/labels/wall.txt").write_text("0 nan 0.5 0.2 0.2\n")
    elif problem == "split_overlap":
        (source / "valid/images/wall.png").write_bytes((source / "train/images/wall.png").read_bytes())
    else:
        (source / "data.yaml").write_text("names: [Jug, Crimp]\n")
    with pytest.raises(ValueError):
        module.prepare_yolo_dataset(source, tmp_path / "prepared")


def test_prepare_supports_nested_zip_and_rejects_path_traversal(tmp_path):
    module = training_module()
    source = export_fixture(tmp_path / "export")
    archive = tmp_path / "export.zip"
    with ZipFile(archive, "w") as stream:
        for path in source.rglob("*"):
            if path.is_file():
                stream.write(path, "nested/" + str(path.relative_to(source)))
    result = module.prepare_yolo_dataset(archive, tmp_path / "prepared")
    assert result["splits"]["train"]["boxes"] == 1
    with ZipFile(tmp_path / "bad.zip", "w") as stream:
        stream.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        module.prepare_yolo_dataset(tmp_path / "bad.zip", tmp_path / "bad-output")
    assert not (tmp_path / "escape.txt").exists()


def test_pipeline_notebook_uses_ready_models_without_training_commands():
    notebook_path = ROOT / "notebooks/DeMoSearch_Pipeline.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
    executable = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for command in ["train-yolo", "prepare-yolo", "download-yolo", "evaluate-yolo"]:
        assert command not in executable
    assert "segment-capture" in executable and "export-wall" in executable
    entrypoint = (ROOT / "scripts/train_yolo26.py").read_text()
    assert "from hold_detection.yolo_training import main" in entrypoint
    assert "def train_yolo26" not in entrypoint


@pytest.mark.parametrize("worker_status,paused,expected", [(0, False, 0), (7, False, 7), (1, True, 75)])
def test_yolo_hpc_launcher_preserves_resume_and_worker_failure(tmp_path, worker_status, paused, expected):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in [
        ("srun", 'printf "%s\\n" "$@"; exit "${FAKE_WORKER_STATUS}"'),
        ("nvidia-smi", 'echo "A100, 40960 MiB"'),
    ]:
        path = binaries / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    output = tmp_path / "run"
    (output / "weights").mkdir(parents=True)
    (output / "weights/last.pt").touch()
    if paused:
        (output / "PAUSED").touch()
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/yolo26-a1002.sbatch")],
        env={**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
             "DEMO_PROJECT": str(ROOT), "DEMO_PYTHON": "/fake/python",
             "DEMO_DATA": str(tmp_path / "data"), "DEMO_RUN": str(output),
             "DEMO_SCRATCH": str(tmp_path / "scratch"), "FAKE_WORKER_STATUS": str(worker_status)},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == expected, result.stderr
    assert "--nproc_per_node=2" in result.stdout
    assert "--resume" in result.stdout


@pytest.mark.skipif(os.environ.get("RUN_YOLO26_SMOKE") != "1", reason="Optional real YOLO26 CPU smoke test")
def test_real_yolo26_can_pause_resume_and_evaluate_on_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "yolo-config"))
    module = training_module()
    source = export_fixture(tmp_path / "export")
    report = module.prepare_yolo_dataset(source, tmp_path / "data")
    output = tmp_path / "run"
    output.mkdir()
    stop_file = tmp_path / "stop-requested"
    stop_file.touch()
    settings = dict(data=report["data_yaml"], output=output, model="yolo26n.yaml",
                    epochs=3, imgsz=64, batch=2, device="cpu", workers=0, stop_file=stop_file)
    with pytest.raises(module.TrainingPaused):
        module.train_yolo26(**settings)
    assert (output / "weights/last.pt").is_file()
    assert not (output / "TRAINING_COMPLETE").exists()
    module.train_yolo26(**settings, resume=True)
    assert (output / "TRAINING_COMPLETE").exists()
    result = module.evaluate_yolo26(report["data_yaml"], output, device="cpu", imgsz=64)
    assert result["split"] == "val"
    assert result["weights_sha256"]
