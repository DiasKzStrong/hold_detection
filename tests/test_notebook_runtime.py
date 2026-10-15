"""Runtime isolation regressions; no model downloads in the unit test suite."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from hold_detection import notebook_runtime as runtime


def fake_installer(calls):
    def install(command, project, log, extra_env=None):
        command = [str(part) for part in command]
        calls.append((command, extra_env))
        if "venv" in command:
            root = Path(command[-1])
            (root / "bin").mkdir(parents=True)
            (root / "bin/python").touch()
            (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    return install


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("sam3", [False, True])
def test_python313_kernel_gets_private_python312_and_pinned_torch(tmp_path, monkeypatch, cuda, sam3):
    calls = []
    monkeypatch.setattr(runtime.sys, "version_info", (3, 13, 5))
    monkeypatch.setattr(runtime.sys, "executable", "/kernel/python313")
    monkeypatch.setattr(runtime, "_uv", lambda *_: Path("/tools/uv"))
    monkeypatch.setattr(runtime, "_gpu_present", lambda: cuda)
    monkeypatch.setattr(runtime, "run_logged", fake_installer(calls))
    monkeypatch.setattr(runtime, "runtime_info", lambda *_: {"cuda_available": cuda})
    legacy = tmp_path / "venv"
    legacy.mkdir()
    (legacy / "keep.txt").write_text("previous user environment")

    python = runtime.prepare_runtime(tmp_path, tmp_path, sam3=sam3)
    flavor = "cu126" if cuda else "cpu"
    assert python == tmp_path / f"runtime-py312-{flavor}-v1/bin/python"
    assert calls[0][0] == ["/tools/uv", "venv", "--no-project", "--python", "3.12", str(python.parent.parent)]
    assert (legacy / "keep.txt").read_text() == "previous user environment"
    installs = [cmd for cmd, _ in calls if "install" in cmd]
    assert installs[0][-1] == f"https://download.pytorch.org/whl/{flavor}"
    for command in installs:
        assert command[command.index("--python") + 1] == str(python)
        assert "torch==2.7.1" in command and "torchvision==0.22.1" in command
        assert "--system" not in command and "--system-site-packages" not in command
    assert any("git+https://github.com/facebookresearch/sam3.git@" in a
               for command in installs for a in command) == sam3
    calls.clear()
    assert runtime.prepare_runtime(tmp_path, tmp_path, sam3=sam3) == python
    assert all("venv" not in cmd for cmd, _ in calls)


def test_inherited_python_paths_cannot_contaminate_private_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/broken-kernel-packages")
    monkeypatch.setenv("PYTHONHOME", "/kernel/python313")
    env = runtime.environment(tmp_path)
    assert env["PYTHONPATH"] == str(tmp_path / "src")
    assert "PYTHONHOME" not in env
    assert env["PYTHONNOUSERSITE"] == "1"


def test_reject_inheriting_venv_without_overwriting_it(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_uv", lambda *_: Path("/tools/uv"))
    root = tmp_path / "runtime-py312-cpu-v1"
    (root / "bin").mkdir(parents=True)
    (root / "bin/python").touch()
    config = root / "pyvenv.cfg"
    config.write_text("include-system-site-packages = true\n")
    with pytest.raises(RuntimeError, match="isolated virtual environment"):
        runtime.prepare_runtime(tmp_path, tmp_path, accelerator="cpu")
    assert config.read_text() == "include-system-site-packages = true\n"


def test_cuda_driver_without_working_torch_cuda_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "_uv", lambda *_: Path("/tools/uv"))
    monkeypatch.setattr(runtime, "run_logged", fake_installer([]))
    monkeypatch.setattr(runtime, "runtime_info", lambda *_: {"cuda_available": False})
    with pytest.raises(RuntimeError, match="CUDA wheels installed but CUDA is unavailable"):
        runtime.prepare_runtime(tmp_path, tmp_path, accelerator="cuda")


@pytest.mark.parametrize("available", [False, True])
def test_driver_detection_does_not_need_torch(monkeypatch, available):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/nvidia-smi" if available else None)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *_, **__: SimpleNamespace(returncode=0, stdout="T4\n"))
    assert runtime._gpu_present() == available


def test_driver_detection_timeout_is_not_a_kernel_import_failure(monkeypatch):
    monkeypatch.setattr(runtime.shutil, "which", lambda _: "/bin/nvidia-smi")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 15)
    monkeypatch.setattr(runtime.subprocess, "run", timeout)
    assert runtime._gpu_present() is False


def test_health_check_runs_in_child_and_reports_import_error(tmp_path, monkeypatch):
    commands = []
    def broken(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stderr="AttributeError: partially initialized module 'torch'")
    monkeypatch.setattr(runtime.subprocess, "run", broken)
    with pytest.raises(RuntimeError, match="isolated model runtime failed"):
        runtime.runtime_info(Path("/private/python"), tmp_path)
    assert commands[0][:3] == ["/private/python", "-I", "-c"]
    assert "torch.library" in commands[0][3] and "torchvision.ops.nms" in commands[0][3]
