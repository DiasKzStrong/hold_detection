"""Build a thin Colab client for the library's YOLO training functions."""

import json
from pathlib import Path

from build_colab_notebook import build_notebook, cell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DeMoSearch_YOLO26_Training.ipynb"


def build_yolo_training_notebook():
    bootstrap = next(c for c in build_notebook()["cells"] if c["id"] == "bootstrap")
    setup = "".join(bootstrap["source"]).replace("demosearch-comparison", "demosearch-yolo26")
    cells = [
        cell("markdown", "intro", """
            # DeMoSearch · YOLO26m training
            This notebook calls ordinary Python modules. Dataset preparation and training live in
            `src/hold_detection/yolo_training.py`; no Python source is exported from this notebook.
            The same functions run through `scripts/train_yolo26.py` on HPC.

            Export a version of HoldClassificatore as **YOLO detection** from Roboflow, then
            upload the ZIP to Drive. Select a GPU runtime for training.
        """),
        cell("code", "bootstrap", setup, hidden=True),
        cell("code", "dataset", """
            from google.colab import drive
            drive.mount('/content/drive')
            DATASET_SOURCE = Path('/content/drive/MyDrive/DeMoSearch/HoldClassificatore.zip')
            PREPARED = WORK / 'datasets' / ('holds_' + uuid.uuid4().hex[:8])
            run_holds('prepare-yolo', DATASET_SOURCE, '--output', PREPARED, log_name='prepare_yolo')
            display(json.loads((PREPARED / 'dataset_report.json').read_text()))
        """),
        cell("code", "settings", """
            MODEL = 'yolo26m.pt'
            EPOCHS = 3
            IMAGE_SIZE = 1024
            BATCH_SIZE = 8
            RESUME = False
            RUN = Path('/content/drive/MyDrive/DeMoSearch/yolo26m_smoke')
        """),
        cell("code", "train", """
            args = ['train-yolo', '--data', PREPARED / 'data.yaml', '--output', RUN,
                    '--model', MODEL, '--epochs', EPOCHS, '--imgsz', IMAGE_SIZE,
                    '--batch', BATCH_SIZE, '--workers', 2, '--device', '0']
            if RESUME: args += ['--resume']
            run_holds(*args, log_name='train_yolo26')
            print('Best weights:', RUN / 'weights/best.pt')
        """),
        cell("code", "evaluate", """
            run_holds('evaluate-yolo', '--data', PREPARED / 'data.yaml', '--output', RUN,
                      '--split', 'val', '--device', '0', '--imgsz', IMAGE_SIZE,
                      log_name='evaluate_yolo26')
            display(json.loads((RUN / 'metrics_val.json').read_text()))
        """),
        cell("markdown", "handoff", """
            Start with 3 epochs, then use a **new run directory** and 100 epochs for a full
            experiment. Resume applies to interrupted runs with the original epoch budget.
            Checkpoints and logs are persistent on Drive. HoldClassificatore has six box-label
            classes, without undercling/sidepull/opening direction. SAM3 supplies masks.

            Full wall analysis (2D photo or 3D photos/video) uses `holds analyze-wall`.
            See `docs/wall-pipeline.md` for the complete SLURM workflow and
            `docs/yolo26-training.md` for training jobs.
        """),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": NOTEBOOK.name, "provenance": []},
    }, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build_yolo_training_notebook(), indent=1, ensure_ascii=False) + "\n")
    print(NOTEBOOK)
