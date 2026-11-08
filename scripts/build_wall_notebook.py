"""Build a thin interactive client for the ordinary Python wall-analysis pipeline."""

import json
from pathlib import Path

from build_colab_notebook import build_notebook, cell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/DeMoSearch_Wall_Analysis.ipynb"


def build_wall_notebook():
    bootstrap = next(c for c in build_notebook()["cells"] if c["id"] == "bootstrap")
    setup = "".join(bootstrap["source"]).replace("demosearch-comparison", "demosearch-wall")
    cells = [
        cell("markdown", "intro", """
            # DeMoSearch · single-photo and multiview wall analysis
            2D: one photo → SAM3 masks / YOLO26 type proposals → scene.json.
            3D: photos/video → COLMAP mesh → masks / multiview correspondence → the same scene.json schema.
            All implementation lives in `src/hold_detection`; this notebook only calls the CLI.
            The route algorithm consumes the exported holds. No beta is generated here.
        """),
        cell("code", "bootstrap", setup, hidden=True),
        cell("code", "sam-setup", """
            import getpass
            if not os.environ.get('HF_TOKEN'):
                os.environ['HF_TOKEN'] = getpass.getpass('Approved Hugging Face read token (hidden): ')
            PYTHON = prepare_runtime(WORK, PROJECT, sam3=True)
            run_holds('fetch-sam3', log_name='fetch_sam3')
        """),
        cell("code", "settings", """
            from google.colab import drive
            drive.mount('/content/drive')
            MODE = '2d'  # @param ["2d", "3d"]
            SOURCE = Path('/content/drive/MyDrive/DeMoSearch/wall.jpg')
            TYPE_WEIGHTS = ''  # Optional fine-tuned six-class YOLO26 best.pt; empty means review types manually.
            OUTPUT = Path('/content/drive/MyDrive/DeMoSearch/analysis_demo')
            COLMAP = 'colmap'  # 3D requires a separately installed CUDA COLMAP executable/container.
            SCALE_M_PER_UNIT = None  # Leave None if no scale calibration is available.
        """),
        cell("code", "analyze", """
            args = ['analyze-wall', SOURCE, '--mode', MODE, '--output', OUTPUT,
                    '--sam-weights', PROJECT / 'models/sam3/sam3.pt', '--device', '0', '--colmap', COLMAP]
            if TYPE_WEIGHTS: args += ['--type-weights', TYPE_WEIGHTS]
            if SCALE_M_PER_UNIT is not None: args += ['--scale-m-per-unit', SCALE_M_PER_UNIT]
            run_holds(*args, log_name='wall_analysis')
        """),
        cell("code", "results", """
            import pandas as pd
            from IPython.display import display
            scene = json.loads((OUTPUT / 'scene.json').read_text())
            print('Mode:', scene['analysis_mode'], '| holds:', len(scene['holds']))
            print('Units:', scene['wall']['units'], '| review:', scene['review'])
            display(pd.DataFrame(scene['holds']).head(30))
            print('Graph algorithm input:', OUTPUT / 'scene.json')
            print('Reconstructed GLB:', scene['wall']['glb_path'])
        """),
        cell("markdown", "hpc", """
            For a full experiment use `scripts/slurm/submit_wall.sh`. It schedules preparation,
            reconstruction (3D only), segmentation and JSON export with afterok dependencies.
            See `docs/wall-pipeline.md`. No HPC job is submitted by opening this notebook.
        """),
    ]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"name": NOTEBOOK.name, "provenance": []},
    }, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build_wall_notebook(), indent=1, ensure_ascii=False) + "\n")
    print(NOTEBOOK)
