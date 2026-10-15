"""Small Colab runtime helpers; heavy installation logs go to files, not the notebook."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SAM3_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
UV_VERSION = "0.12.23"
TORCH_PACKAGES = ["torch==2.7.1", "torchvision==0.22.1"]


def environment(project: Path):
    result = os.environ.copy()
    result.pop("PYTHONHOME", None)
    result["PYTHONPATH"] = str(project / "src")
    result["PYTHONNOUSERSITE"] = "1"
    result["HF_HUB_DISABLE_XET"] = "1"
    result["MPLBACKEND"] = "Agg"
    return result


def run_logged(command, project: Path, log: Path, extra_env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = environment(project)
    env.update(extra_env or {})
    with log.open("w") as stream:
        result = subprocess.run([str(p) for p in command], cwd=project,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-18:])
        raise RuntimeError(f"Command failed ({result.returncode}); log: {log}\n{tail}")
    return result


def _uv(work: Path, project: Path):
    installed = shutil.which("uv")
    if installed:
        return Path(installed)
    tools = work / "bootstrap-tools"
    binary = tools / "bin/uv"
    if not binary.is_file():
        run_logged([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-deps", "--target", tools, f"uv=={UV_VERSION}"],
                   project, work / "logs/install_uv.log")
    if not binary.is_file():
        raise RuntimeError(f"uv installation did not create {binary}; inspect logs/install_uv.log")
    return binary


def _gpu_present():
    """Check the driver without importing the notebook kernel's PyTorch."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        result = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def prepare_runtime(work: Path, project: Path, sam3=False, accelerator="auto"):
    """Use Python 3.12 and private wheels even when Colab runs Python 3.13.

    Never import or repair the kernel's torch, and never inherit its site packages.
    The legacy work/venv directory is left untouched.
    """
    work, project = Path(work).resolve(), Path(project).resolve()
    if accelerator not in {"auto", "cpu", "cuda"}:
        raise ValueError("accelerator must be auto, cpu or cuda")
    cuda = _gpu_present() if accelerator == "auto" else accelerator == "cuda"
    flavor = "cu126" if cuda else "cpu"
    runtime = work / f"runtime-py312-{flavor}-v1"
    python = runtime / "bin/python"
    uv = _uv(work, project)
    tool_env = {"UV_CACHE_DIR": str(work / "cache/uv"),
                "UV_PYTHON_INSTALL_DIR": str(work / "python"), "UV_PYTHON_DOWNLOADS": "automatic"}
    if not python.is_file():
        if runtime.exists() and any(runtime.iterdir()):
            raise RuntimeError(f"Incomplete environment at {runtime}; choose a new WORK directory to preserve it.")
        interpreter = sys.executable if sys.version_info[:2] == (3, 12) else "3.12"
        run_logged([uv, "venv", "--no-project", "--python", interpreter, runtime],
                   project, work / "logs/create_runtime.log", extra_env=tool_env)
    config = runtime / "pyvenv.cfg"
    if not config.is_file() or "include-system-site-packages = false" not in config.read_text().lower():
        raise RuntimeError(f"Expected an isolated virtual environment at {runtime}; choose another WORK directory.")
    index = f"https://download.pytorch.org/whl/{flavor}"
    install = [uv, "pip", "install", "--python", python]
    print(f"Preparing isolated Python 3.12 · {flavor}; first installation may take several minutes.")
    run_logged([*install, *TORCH_PACKAGES, "--index-url", index],
               project, work / "logs/install_torch.log", extra_env=tool_env)
    common = ["numpy==1.26.4", "opencv-python==4.11.0.86", "ultralytics==8.3.249",
              "pycocotools==2.0.11", "pillow>=11,<13", "matplotlib>=3.8,<4"]
    run_logged([*install, *TORCH_PACKAGES, *common], project,
               work / "logs/install_baseline.log", extra_env=tool_env)
    if sam3:
        dependencies = ["setuptools>=77,<81", "wheel>=0.43,<1", "einops==0.8.1", "decord==0.6.0",
                        "psutil>=5.9", "huggingface-hub==2.2.0", "timm==1.0.30", "iopath==0.1.10",
                        "ftfy==6.1.1", "regex", "requests"]
        run_logged([*install, *TORCH_PACKAGES, *common, *dependencies], project,
                   work / "logs/install_sam_deps.log", extra_env=tool_env)
        run_logged([*install, *TORCH_PACKAGES, *common, "--no-build-isolation",
                    f"git+https://github.com/facebookresearch/sam3.git@{SAM3_REVISION}"],
                   project, work / "logs/install_sam.log", extra_env=tool_env)
    info = runtime_info(python, project)
    if cuda and not info["cuda_available"]:
        raise RuntimeError("CUDA wheels installed but CUDA is unavailable. Select a Colab GPU runtime "
                           "and check the NVIDIA driver; the notebook kernel's torch was not modified.")
    return python


def run_cli(python: Path, project: Path, arguments, log: Path):
    return run_logged([python, "-m", "hold_detection.cli", *arguments], project, log)


def runtime_info(python, project):
    probe = """
import json,sys,torch,numpy,torchvision
from importlib.metadata import version,PackageNotFoundError
assert sys.version_info[:2] == (3, 12), 'The model runtime must use Python 3.12'
assert torch.__version__.split('+')[0] == '2.7.1', 'Unexpected PyTorch version'
assert torchvision.__version__.split('+')[0] == '0.22.1', 'Unexpected torchvision version'
assert hasattr(torch.library, 'custom_op'), 'PyTorch custom_op is unavailable'
torch.from_numpy(numpy.zeros(1, dtype=numpy.float32))
torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5)
cuda_available = torch.cuda.is_available()
if cuda_available:
    torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]], device='cuda'),
                        torch.tensor([1.], device='cuda'), 0.5)
    torch.cuda.synchronize()
packages={}
for p in ['torchvision','ultralytics','sam3','pycocotools','triton']:
    try: packages[p]=version(p)
    except PackageNotFoundError: packages[p]=None
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,
                  'numpy':numpy.__version__,'cuda':torch.version.cuda,
                  'cuda_available':cuda_available,
                  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  'packages':packages}))
"""
    result = subprocess.run([str(python), "-I", "-c", probe], cwd=project, env=environment(project),
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("The isolated model runtime failed its Torch/torchvision health check:\n"
                           + result.stderr[-2500:])
    return json.loads(result.stdout)
