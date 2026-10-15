"""Generate the readable YOLO-versus-SAM-3 Colab experiment notebook."""

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "DeMoSearch_SAM3_Colab.ipynb"


def cell(kind, name, source, hidden=False):
    metadata = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}
    item = {"cell_type": kind, "id": name, "metadata": metadata,
            "source": dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def build_notebook():
    cells = [
        cell("markdown", "intro", """
            # DeMoSearch · YOLO vs SAM 3
            **One photo set → two models → visual comparison → measurable results.**

            | Experiment | Localization | Hold type |
            |:---|:---|:---|
            | **YOLO11s baseline** | Bounding boxes | Six classes from the detector head |
            | **SAM 3 + classifier** | Text-prompted instance masks | Legacy CNN or trained ResNet18/SigLIP |

            Start with 3 frames. Select **Runtime → Change runtime type → GPU**.
            Model inference uses a private Python 3.12 environment, even with a Python 3.13 kernel.
            YOLO also works on CPU without Hugging Face. SAM requires a GPU and approved
            access to [facebook/sam3](https://huggingface.co/facebook/sam3).

            **Results:** original / YOLO / SAM side by side, frame and threshold controls,
            crop grids, latency and score plots. With COCO annotations: precision/recall/F1,
            box AP, mask AP/IoU and type confusion matrices. **No ground truth, no quality claims.**
        """),
        cell("markdown", "setup-heading", """
            ## 1 · Environment
            Setup cells are collapsed. A private Python 3.12 environment installs pinned
            PyTorch/torchvision wheels without importing or modifying Colab's system Torch.
            The first installation may take several minutes. Source code
            comes from our public GitHub repository, without base64 payloads; the exact
            commit is recorded with the results.
        """),
        cell("code", "bootstrap", """
            # @title Set up the environment (once)
            import json, os, subprocess, sys, uuid
            from pathlib import Path
            from datetime import datetime, timezone

            CODE_REF = 'main'  # @param {type: "string"}
            WORK = Path('/content/demosearch-comparison')
            PROJECT = WORK / 'project'
            WORK.mkdir(parents=True, exist_ok=True)
            repository = 'https://github.com/DiasKzStrong/hold_detection.git'
            if not PROJECT.exists():
                subprocess.run(['git', 'clone', '--quiet', '--depth', '1', repository, str(PROJECT)], check=True)
            if not (PROJECT / '.git').is_dir():
                raise RuntimeError('The project directory contains unrelated files; choose another WORK path.')
            dirty = subprocess.check_output(['git', '-C', str(PROJECT), 'status', '--porcelain'], text=True)
            if dirty.strip():
                raise RuntimeError('Save local project changes before updating; they will not be overwritten.')
            subprocess.run(['git', '-C', str(PROJECT), 'fetch', '--quiet', 'origin', CODE_REF], check=True)
            CODE_SHA = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'FETCH_HEAD'], text=True).strip()
            previous = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
            if previous != CODE_SHA and 'hold_detection' in sys.modules:
                raise RuntimeError('Source code has changed: restart the runtime to avoid mixing module versions.')
            subprocess.run(['git', '-C', str(PROJECT), 'checkout', '--quiet', '--detach', CODE_SHA], check=True)
            if str(PROJECT / 'src') not in sys.path: sys.path.insert(0, str(PROJECT / 'src'))
            from hold_detection.notebook_runtime import prepare_runtime, run_cli, runtime_info
            PYTHON = prepare_runtime(WORK, PROJECT)
            MODEL_ENV = runtime_info(PYTHON, PROJECT)

            def run_holds(*arguments, log_name='command'):
                return run_cli(PYTHON, PROJECT, list(arguments), WORK / 'logs' / (log_name + '.log'))

            print('Ready ·', MODEL_ENV['gpu'] if MODEL_ENV['cuda_available'] else 'CPU baseline')
            print('Kernel Python:', sys.version.split()[0], '| model Python:', MODEL_ENV['python'])
            print('Code commit:', CODE_SHA, '| installation logs:', WORK / 'logs')
        """, hidden=True),
        cell("markdown", "data-heading", """
            ## 2 · Data and settings
            Upload JPEG/PNG images, a photo ZIP, or `wallA_1_raw_data.zip`. For Polycam,
            RGB frames are selected instead of depth PNGs. Use Google Drive for larger datasets.
        """),
        cell("code", "upload-inputs", """
            # @title Load photos
            from google.colab import files, drive
            from hold_detection.colab_support import prepare_inputs

            INPUT_MODE = 'upload'  # @param ["upload", "drive"]
            DRIVE_INPUT = '/content/drive/MyDrive/DeMoSearch/wallA_1_raw_data.zip'  # @param {type: "string"}
            data_id = uuid.uuid4().hex[:10]
            if INPUT_MODE == 'drive':
                drive.mount('/content/drive')
                selected = [Path(DRIVE_INPUT)]
            else:
                upload = WORK / 'uploads' / data_id
                upload.mkdir(parents=True, exist_ok=False)
                previous_directory = Path.cwd()
                try:
                    os.chdir(upload)
                    uploaded = files.upload()
                finally:
                    os.chdir(previous_directory)
                if not uploaded or any(Path(n).name != n for n in uploaded):
                    raise ValueError('Select photos or one ZIP with filenames that contain no directory paths.')
                selected = [upload / name for name in uploaded]
                del uploaded
            prepared = prepare_inputs(selected, WORK / 'data' / data_id)
            SOURCE = prepared.source
            print(f'{prepared.image_count} RGB images · {prepared.kind}')
        """, hidden=True),
        cell("code", "settings", """
            MAX_IMAGES = 3       # @param {type: "integer"}
            ROTATE = -1          # @param {type: "integer"}
            RUN_SAM3 = True      # @param {type: "boolean"}
            PROMPT = 'climbing hold'  # @param {type: "string"}
            CANDIDATE_FLOOR = 0.05    # @param {type: "number"}
            YOLO_THRESHOLD = 0.35    # @param {type: "number"}
            SAM_THRESHOLD = 0.50     # @param {type: "number"}
            MIN_TYPE_SCORE = 0.50    # @param {type: "number"}
            CLASSIFIER = 'cnn'  # @param ["cnn", "trained", "none"]
            CLASSIFIER_WEIGHTS = ''  # @param {type: "string"}

            if CLASSIFIER == 'trained' and not Path(CLASSIFIER_WEIGHTS).is_file():
                raise ValueError('Set CLASSIFIER_WEIGHTS to your trained ResNet18/SigLIP best.pt artifact.')

            if MAX_IMAGES < 0 or ROTATE not in (-1, 0, 90, 180, 270):
                raise ValueError('MAX_IMAGES must be >= 0; ROTATE must be -1, 0, 90, 180 or 270.')
            if not all(0 <= v <= 1 for v in [CANDIDATE_FLOOR, YOLO_THRESHOLD, SAM_THRESHOLD, MIN_TYPE_SCORE]):
                raise ValueError('Scores must be in [0, 1].')
            if CANDIDATE_FLOOR > min(YOLO_THRESHOLD, SAM_THRESHOLD):
                raise ValueError('The candidate floor must not exceed the viewing thresholds.')
            rotation = (90 if prepared.kind == 'polycam' else 0) if ROTATE == -1 else ROTATE
            experiment_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            EXPERIMENT = PROJECT / 'runs' / ('comparison_' + experiment_id)
            EXPERIMENT.mkdir(parents=True, exist_ok=False)
            RUNS, THRESHOLDS = {}, {}
            print('Preview rotation:', rotation, '| images:', MAX_IMAGES or 'all')
        """),
        cell("markdown", "protocol-note", """
            `MAX_IMAGES=0` selects all frames. `ROTATE=-1` means 90° for our Polycam scan
            and 0° otherwise; check the preview. A low `CANDIDATE_FLOOR` preserves proposals
            for AP, while viewing thresholds hide weak predictions without rerunning inference.
            AP is limited by this floor. YOLO and SAM scores **are not calibrated against each other**.
        """),
        cell("code", "preview", """
            import matplotlib.pyplot as plt
            from PIL import Image
            from IPython.display import display, HTML
            from hold_detection.pipeline import collect_images

            image_paths = collect_images(SOURCE)
            selected_paths = image_paths[:MAX_IMAGES] if MAX_IMAGES else image_paths
            fig, axes = plt.subplots(1, min(4, len(selected_paths)), figsize=(15, 4), squeeze=False)
            for ax, path in zip(axes[0], selected_paths):
                with Image.open(path) as image:
                    ax.imshow(image.convert('RGB').rotate(-rotation, expand=True))
                ax.set_title(path.name, fontsize=9)
                ax.axis('off')
            plt.tight_layout(); plt.show()
        """),
        cell("markdown", "baseline-heading", """
            ## 3 · YOLO11s baseline
            Detection and hold types come from one model. This section does not require HF_TOKEN.
            Model latency is measured separately from image export; weight loading is excluded,
            and the first frame includes warmup. GPU timers are synchronized.
        """),
        cell("code", "baseline", """
            print('Running YOLO baseline…')
            run_holds('fetch-models', log_name='fetch_yolo')
            yolo_path = EXPERIMENT / ('yolo_' + uuid.uuid4().hex[:6])
            args = ['predict', SOURCE, '--output', yolo_path, '--rotate', rotation,
                    '--conf', CANDIDATE_FLOOR, '--min-type-score', MIN_TYPE_SCORE, '--max-det', 500]
            if MAX_IMAGES: args += ['--limit', MAX_IMAGES]
            run_holds(*args, log_name='yolo_inference')
            RUNS['YOLO11s'] = yolo_path
            THRESHOLDS['YOLO11s'] = YOLO_THRESHOLD
            from hold_detection.visualization import plot_comparison
            display(plot_comparison(RUNS, thresholds=THRESHOLDS)); plt.close('all')
        """),
        cell("markdown", "sam-heading", """
            ## 4 · SAM 3 + crop classifier
            SAM proposes masks independently from a text prompt; it does not receive YOLO boxes.
            Localization is compared separately from type classification quality.
            With `RUN_SAM3=False` or no GPU, the full YOLO baseline remains available.
            For a trained ResNet18/SigLIP, set `CLASSIFIER='trained'` and provide `best.pt`.
            Use [the training notebook](https://colab.research.google.com/github/DiasKzStrong/hold_detection/blob/main/notebooks/DeMoSearch_Classifier_Training.ipynb)
            or the NYU SLURM jobs to produce that artifact.
        """),
        cell("code", "sam-setup", """
            # @title Set up SAM 3 and checkpoint access
            import getpass
            SAM_READY = False
            if RUN_SAM3 and MODEL_ENV['cuda_available']:
                print('Preparing SAM 3; installation and checkpoint download may take several minutes…')
                from google.colab import userdata
                try:
                    if not os.environ.get('HF_TOKEN'):
                        try:
                            token = userdata.get('HF_TOKEN')
                        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                            token = getpass.getpass('Hugging Face read token (hidden): ').strip()
                        if not token: raise ValueError('A token for an account with approved access to facebook/sam3 is required.')
                        os.environ['HF_TOKEN'] = token
                        del token
                    PYTHON = prepare_runtime(WORK, PROJECT, sam3=True, training=CLASSIFIER == 'trained')
                    run_holds('fetch-sam3', log_name='fetch_sam3')
                    SAM_READY = True
                    print('SAM 3 ready. Token value is not displayed or saved.')
                except Exception as error:
                    print('SAM 3 NOT READY:', error)
                    print('YOLO results remain available; fix access/setup and rerun this cell.')
            else:
                print('SAM 3 not run: disabled or no CUDA GPU. No SAM predictions will be fabricated.')
        """, hidden=True),
        cell("code", "sam3-run", """
            RUNS.pop('SAM 3', None)
            THRESHOLDS.pop('SAM 3', None)
            if SAM_READY:
                sam_path = EXPERIMENT / ('sam3_' + uuid.uuid4().hex[:6])
                args = ['sam3', SOURCE, '--output', sam_path, '--rotate', rotation,
                        '--prompt', PROMPT, '--conf', CANDIDATE_FLOOR, '--max-det', 500,
                        '--classifier', CLASSIFIER, '--min-type-score', MIN_TYPE_SCORE]
                if CLASSIFIER == 'trained': args += ['--classifier-weights', CLASSIFIER_WEIGHTS]
                if MAX_IMAGES: args += ['--limit', MAX_IMAGES]
                print('Running SAM 3 segmentation +', CLASSIFIER, '…')
                try:
                    run_holds(*args, log_name='sam3_inference')
                    RUNS['SAM 3'] = sam_path
                    THRESHOLDS['SAM 3'] = SAM_THRESHOLD
                    print('SAM 3 finished on the same frames as YOLO.')
                except RuntimeError as error:
                    print('SAM 3 inference failed:', error)
                    print('YOLO remains available. Fix the error shown above and rerun this cell.')
                    if 'cuda' in str(error).lower() and 'out of memory' in str(error).lower():
                        print('For CUDA out-of-memory errors, try a larger GPU.')
        """),
        cell("markdown", "viewer-heading", """
            ## 5 · Frame-by-frame comparison
            Select a frame, adjust thresholds, and toggle masks/boxes. ID labels are off
            by default to avoid obscuring holds. Inspect types and scores in the crops below.
        """),
        cell("code", "viewer", """
            from google.colab import output
            output.enable_custom_widget_manager()
            from hold_detection.visualization import ComparisonViewer, plot_crop_grid

            viewer = ComparisonViewer(RUNS, THRESHOLDS)
            display(viewer.widget)
        """),
        cell("code", "crop-grid", """
            import ipywidgets as widgets
            crop_model = widgets.Dropdown(options=list(RUNS), value=list(RUNS)[-1], description='Crops:')
            crop_output = widgets.Output()

            def show_crops(*_):
                with crop_output:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    fig = plot_crop_grid(RUNS[crop_model.value], viewer.frame.value,
                                         viewer.thresholds[crop_model.value], maximum=18)
                    display(fig); plt.close(fig)

            crop_model.observe(show_crops, names='value')
            viewer.frame.observe(show_crops, names='value')
            for slider in viewer.sliders.values(): slider.observe(show_crops, names='value')
            display(widgets.VBox([crop_model, crop_output])); show_crops()
        """),
        cell("markdown", "gt-heading", """
            ## 6 · Ground truth — optional
            Upload a **COCO instance JSON** from CVAT/Roboflow for quality metrics.
            Annotate all visible holds/volumes on evaluated frames; partial annotations
            produce spurious false positives. Image filenames must match the uploaded photos.

            The `hold` category enables detection metrics; `crimp/jug/pinch/pocket/sloper/volume`
            also enable type metrics. Polygon masks or COCO RLE are required for mask AP.
            The YOLO baseline has no masks: its mask AP is **N/A**, not 0.

            `source` uses annotations in original-image pixels; rotation is applied automatically.
            `processed` uses annotations on already-rotated images. Without GT, only diagnostics are available.
        """),
        cell("code", "gt-upload", """
            USE_GROUND_TRUTH = False  # @param {type: "boolean"}
            GT_COORDINATES = 'source'  # @param ["source", "processed"]
            GT_FILE = None
            if USE_GROUND_TRUTH:
                annotation_dir = WORK / 'annotations' / uuid.uuid4().hex[:8]
                annotation_dir.mkdir(parents=True, exist_ok=False)
                previous_directory = Path.cwd()
                try:
                    os.chdir(annotation_dir)
                    uploaded = files.upload()
                finally:
                    os.chdir(previous_directory)
                if len(uploaded) != 1 or not next(iter(uploaded)).lower().endswith('.json'):
                    raise ValueError('Select one COCO JSON file.')
                GT_FILE = annotation_dir / next(iter(uploaded))
                del uploaded
            print('Quality metrics:', 'COCO annotations supplied' if GT_FILE else 'unavailable — no ground truth')
        """),
        cell("markdown", "metrics-heading", """
            ## 7 · Metrics
            Thresholds are read from the viewer **when this cell runs**. Rerun it after
            changing the sliders. AP uses all saved proposals, while P/R/F1 use the
            selected thresholds. Matching is class-agnostic, one-to-one, IoU ≥ 0.50, maxDets=500.
            AP50–95 uses the COCO IoU grid, but the limit of 500 differs from the standard 100.

            Type accuracy/F1 are computed on matched GT; coverage and the fraction of all
            typed GT that are both detected **and** correctly classified are shown separately.
            This prevents high conditional accuracy from hiding missed holds.
        """),
        cell("code", "metrics", """
            import pandas as pd
            from hold_detection.visualization import metric_rows, plot_diagnostics, plot_quality, plot_type_confusions

            REPORT_DIR = EXPERIMENT / ('report_' + uuid.uuid4().hex[:6])
            args = ['benchmark', '--output', REPORT_DIR, '--gt-coordinates', GT_COORDINATES, '--max-det', 500]
            for name, path in RUNS.items():
                args += ['--run', name, path, '--threshold', name, viewer.thresholds[name]]
            if GT_FILE: args += ['--annotations', GT_FILE]
            run_holds(*args, log_name='benchmark')
            report = json.loads((REPORT_DIR / 'comparison.json').read_text())
            table = pd.DataFrame(metric_rows(report))

            def show_table(columns):
                display(table[columns].style.format(precision=3, na_rep='—').hide(axis='index'))

            show_table(['Model', 'Images', 'Candidates', 'Threshold', 'Model p50 ms', 'Model p95 ms'])
            display(plot_diagnostics(RUNS, report)); plt.close('all')
            if GT_FILE:
                show_table(['Model', 'Precision', 'Recall', 'F1', 'Box AP50', 'Box AP50–95', 'Mask AP50–95', 'Mask IoU · matched'])
                display(plot_quality(report)); plt.close('all')
                show_table(['Model', 'Type acc · matched', 'Type macro-F1 · matched', 'Type coverage', 'Correct type recall · all GT'])
                confusion = plot_type_confusions(report)
                if confusion is not None: display(confusion); plt.close(confusion)
            else:
                display(HTML('<p style="padding:14px;background:#f1f5f9;border-radius:8px">'
                             'Quality metrics unavailable. Upload COCO ground truth in section 6, then rerun section 7.</p>'))
        """),
        cell("code", "errors", """
            if GT_FILE:
                quality = {row['model']: row['quality'] for row in report['models']}
                evaluated_thresholds = {row['model']: row['threshold'] for row in report['models']}
                display(plot_comparison(RUNS, viewer.frame.value, evaluated_thresholds, quality=quality))
                plt.close('all')
                print('Green = TP · red = FP · dashed orange = missed GT. Types are evaluated separately.')
        """),
        cell("markdown", "export-heading", """
            ## 8 · Download the experiment
            The ZIP contains available model galleries, masks/crops, comparison figures,
            metrics.csv, comparison.json and runtime versions. Tokens and weights are excluded.
            Extract the entire folder and open `report_…/index.html`.
        """),
        cell("code", "export", """
            from zipfile import ZipFile, ZIP_DEFLATED

            session = {'source_commit': CODE_SHA, 'sam3_requested': RUN_SAM3, 'sam3_ready': SAM_READY,
                       'runtime': runtime_info(PYTHON, PROJECT),
                       'runs': {k: str(v.relative_to(EXPERIMENT)) for k, v in RUNS.items()},
                       'latest_report': str(REPORT_DIR.relative_to(EXPERIMENT))}
            (EXPERIMENT / 'session.json').write_text(json.dumps(session, indent=2))
            RESULT_ZIP = WORK / (EXPERIMENT.name + '_' + uuid.uuid4().hex[:6] + '.zip')
            with ZipFile(RESULT_ZIP, 'x', compression=ZIP_DEFLATED) as archive:
                for path in sorted(EXPERIMENT.rglob('*')):
                    if path.is_symlink(): raise ValueError('Result symlinks are not bundled')
                    if path.is_file(): archive.write(path, EXPERIMENT.name + '/' + path.relative_to(EXPERIMENT).as_posix())
            print('Open after extraction:', REPORT_DIR.relative_to(EXPERIMENT) / 'index.html')
            files.download(str(RESULT_ZIP))
        """),
        cell("markdown", "limits", """
            ---
            **Experiment limitations.** The CNN is a baseline: 69/100 correct on the
            upstream author's sample crops, not on our gym. Grip type may not be visible
            from a particular viewpoint. Both models may mistake route tags for holds.
            Do not randomly split neighboring frames from one scan between train/test.
            Real SAM GPU inference has not been verified in the development environment;
            it runs in this Colab runtime.
        """),
    ]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "accelerator": "GPU",
                         "colab": {"name": "DeMoSearch · YOLO vs SAM 3", "provenance": []},
                         "demosearch": {"experiment_version": 2, "gpu_inference_validated": False}},
            "cells": cells}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
