import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def test_training_notebook_is_readable_valid_and_deterministic(monkeypatch):
    path = ROOT / "notebooks/DeMoSearch_Classifier_Training.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=cell.id)
            assert cell.outputs == [] and cell.execution_count is None
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "training_builder", ROOT / "scripts/build_training_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    assert json.loads(path.read_text()) == builder.build_training_notebook()
    bootstrap = next(c.source for c in notebook.cells if c.id == "bootstrap")
    assert "training=True" in bootstrap and "import torch" not in bootstrap
    head = next(c.source for c in notebook.cells if c.id == "siglip-head")
    assert "'cpu'" in head


def test_nyu_jobs_use_requested_cloud_account_two_a100s_and_restart():
    folder = ROOT / "scripts/slurm"
    for name in ["a1002.sbatch", "cpu.sbatch", "container_python.sh"]:
        subprocess.run(["bash", "-n", str(folder / name)], check=True)
    gpu = (folder / "a1002.sbatch").read_text()
    assert "--account=csci_ga_2271-2026fa" in gpu
    assert "--partition=c24m170-a100-2" in gpu and "--gres=gpu:2" in gpu
    assert "--gpus 2" in gpu and "--resume" in gpu and "--requeue" in gpu
    cpu = (folder / "cpu.sbatch").read_text()
    assert "--partition=n2c48m24" in cpu and "--gres" not in cpu
    assert "--device cpu" in cpu and "--requeue" in cpu
    for text in [gpu, cpu]:
        assert "/scratch/" in text and "--signal=USR1@120" in text
        assert "--partition=torch" not in text


def test_a100_job_dry_run_uses_two_gpus_and_automatic_resume(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, body in [
        ("srun", 'printf "%s\\n" "$@"'),
        ("nvidia-smi", 'echo "A100, 40960 MiB"'),
    ]:
        script = binaries / name
        script.write_text("#!/bin/sh\n" + body + "\n")
        script.chmod(0o755)
    output = tmp_path / "run"
    output.mkdir()
    (output / "last.pt").touch()
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "DEMO_PROJECT": str(ROOT),
        "DEMO_DATA": str(tmp_path / "crops"),
        "DEMO_PYTHON": "/fake/python",
        "DEMO_RUN": str(output),
        "DEMO_SCRATCH": str(tmp_path / "scratch"),
        "DEMO_STAGE": "resnet18",
    }
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/slurm/a1002.sbatch")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--resume" in result.stdout
    assert "--gpus\n2\n" in result.stdout and "--batch-size\n256\n" in result.stdout
