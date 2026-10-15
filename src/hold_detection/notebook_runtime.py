"""Small Colab runtime helpers; heavy installation logs go to files, not the notebook."""

import json
import os
from pathlib import Path
import subprocess
import venv

SAM3_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


def environment(project: Path):
    result = os.environ.copy()
    result["PYTHONPATH"] = str(project / "src") + os.pathsep + result.get("PYTHONPATH", "")
    result["HF_HUB_DISABLE_XET"] = "1"
    result["MPLBACKEND"] = "Agg"
    return result


def run_logged(command, project: Path, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        result = subprocess.run([str(p) for p in command], cwd=project,
                                env=environment(project), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-18:])
        raise RuntimeError(f"Command failed ({result.returncode}); log: {log}\n{tail}")
    return result


def prepare_runtime(work: Path, project: Path, sam3=False):
    runtime = work / "venv"
    if not (runtime / "bin/python").is_file():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(runtime)
    python = runtime / "bin/python"
    common = ["numpy==1.26.4", "opencv-python==4.11.0.86", "ultralytics==8.3.249",
              "pycocotools==2.0.11", "pillow>=11,<13", "matplotlib>=3.8,<4"]
    if not sam3:
        run_logged([python, "-m", "pip", "install", *common], project, work / "logs/install_baseline.log")
    else:
        dependencies = ["setuptools>=77,<81", "wheel>=0.43,<1", "einops==0.8.1", "decord==0.6.0",
                        "psutil>=5.9", "huggingface-hub==2.2.0", "timm==1.0.30", "iopath==0.1.10",
                        "ftfy==6.1.1", "regex", "requests"]
        run_logged([python, "-m", "pip", "install", *dependencies], project, work / "logs/install_sam_deps.log")
        run_logged([python, "-m", "pip", "install", "--no-build-isolation",
                    f"git+https://github.com/facebookresearch/sam3.git@{SAM3_REVISION}"],
                   project, work / "logs/install_sam.log")
    return python


def run_cli(python: Path, project: Path, arguments, log: Path):
    return run_logged([python, "-m", "hold_detection.cli", *arguments], project, log)


def runtime_info(python, project):
    probe = """
import json,sys,torch,numpy
from importlib.metadata import version,PackageNotFoundError
packages={}
for p in ['torchvision','ultralytics','sam3','pycocotools','triton']:
    try: packages[p]=version(p)
    except PackageNotFoundError: packages[p]=None
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,
                  'numpy':numpy.__version__,'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  'packages':packages}))
"""
    result = subprocess.run([str(python), "-c", probe], cwd=project, env=environment(project),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout)
