"""Generate the English Colab companion for supervised crop classifier training."""

import json
from pathlib import Path

from build_colab_notebook import build_notebook, cell

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/DeMoSearch_Classifier_Training.ipynb"


def build_training_notebook():
    bootstrap = next(c for c in build_notebook()["cells"] if c["id"] == "bootstrap")
    source = (
        "".join(bootstrap["source"])
        .replace(
            "prepare_runtime(WORK, PROJECT)",
            "prepare_runtime(WORK, PROJECT, training=True)",
        )
        .replace("demosearch-comparison", "demosearch-classifiers")
    )
    cells = [
        cell(
            "markdown",
            "intro",
            """
            # DeMoSearch · ResNet18 vs SigLIP linear probe
            Train grip classifiers, not detectors, from labeled YOLO/COCO wall images.

            | Experiment | Encoder | Trainable parameters |
            |:---|:---|:---|
            | ResNet18 | ImageNet pretrained | Full network + six-class head |
            | SigLIP | `google/siglip-base-patch16-224` | Linear head; encoder frozen |

            **Unlabeled Polycam photos are for inference, not supervised training.**
            Download a labeled detection export of
            [HoldClassificatore_v3_mix](https://universe.roboflow.com/climbing-holds-classification/holdclassificatore_v3_mix).
            Export YOLO detection or COCO boxes, retain attribution and the version/split information.

            For limited Colab quota, use this notebook to check data and a short run.
            Full NYU training: `scripts/slurm/a1002.sbatch`, partition `c24m170-a100-2`.
            Preparation and cached linear-head training have CPU job templates too.
        """,
        ),
        cell("code", "bootstrap", source, hidden=True),
        cell(
            "markdown",
            "data-heading",
            """
            ## 1 · Labeled dataset
            Upload a labeled ZIP or choose its Drive path. An export contains
            `train/images`, `train/labels`, `valid/images`, `valid/labels` and `data.yaml`,
            or per-split `_annotations.coco.json`. Kaggle `Final_Dataset` is also supported.
            If Kaggle has no YAML, supply class names in annotation ID order below.

            Preparation preserves source splits. Exact duplicate pixels crossing splits
            are rejected. Filename-family warnings are not proof of independent splits:
            audit mirrors/resizes/neighboring frames and physical wall/scan groups yourself.
        """,
        ),
        cell(
            "code",
            "dataset-input",
            """
            from google.colab import files, drive
            INPUT_MODE = 'drive'  # @param ["drive", "upload"]
            DRIVE_DATASET = '/content/drive/MyDrive/DeMoSearch-data/holds-labeled.zip'  # @param {type: "string"}
            EXISTING_CROPS = ''  # @param {type: "string"}
            CLASS_NAMES = ''  # @param {type: "string"}
            GROUPS_CSV = ''  # @param {type: "string"}
            if INPUT_MODE == 'drive' or EXISTING_CROPS.startswith('/content/drive'):
                drive.mount('/content/drive')
            if EXISTING_CROPS:
                CROP_DATA = Path(EXISTING_CROPS)
            else:
                if INPUT_MODE == 'upload':
                    target = WORK / 'uploads' / uuid.uuid4().hex[:8]
                    target.mkdir(parents=True, exist_ok=False)
                    previous = Path.cwd()
                    try:
                        os.chdir(target); uploaded = files.upload()
                    finally:
                        os.chdir(previous)
                    if len(uploaded) != 1 or any(Path(n).name != n for n in uploaded):
                        raise ValueError('Select one labeled dataset ZIP with a simple filename.')
                    source = target / next(iter(uploaded)); del uploaded
                else:
                    source = Path(DRIVE_DATASET)
                if not source.exists():
                    raise FileNotFoundError(f'Dataset not found: {source}. Use Files → Copy path.')
                CROP_DATA = WORK / 'data' / ('crops_' + uuid.uuid4().hex[:8])
                args = ['prepare-crops', source, '--output', CROP_DATA]
                if CLASS_NAMES: args += ['--class-names', *CLASS_NAMES.replace(',', ' ').split()]
                if GROUPS_CSV: args += ['--groups', GROUPS_CSV]
                run_holds(*args, log_name='prepare_crops')
            from hold_detection.crop_dataset import load_crops
            crop_manifest = load_crops(CROP_DATA)
            print('Prepared crops:', CROP_DATA)
            print('Reuse this exact CROP_DATA path when resuming training.')
        """,
        ),
        cell(
            "code",
            "crop-preview",
            """
            import pandas as pd
            import matplotlib.pyplot as plt
            from PIL import Image
            from IPython.display import display
            display(pd.DataFrame(crop_manifest['class_counts']).rename_axis('Class'))
            warnings = crop_manifest['suspected_cross_split_filename_families']
            print('Possible filename-family leakage:', len(warnings), '| examples:', warnings[:10])
            fig, axes = plt.subplots(1,6,figsize=(15,3))
            for index, ax in enumerate(axes):
                sample = next((r for r in crop_manifest['samples'] if r['split']=='train' and r['label']==index), None)
                if sample:
                    with Image.open(CROP_DATA / sample['path']) as crop: ax.imshow(crop.copy())
                ax.set_title(crop_manifest['classes'][index]); ax.axis('off')
            plt.tight_layout(); plt.show()
        """,
        ),
        cell(
            "markdown",
            "settings-heading",
            """
            ## 2 · Budget and run settings
            SigLIP extracts frozen features once; rerunning extraction skips cached batches.
            A linear head can then train on CPU. ResNet18 fine-tunes its entire encoder.
            Batch size is **global**, not per GPU. Colab normally has one visible GPU.

            To resume: set the same crop dataset, RUN_DIRECTORY, epochs, batch size and seed.
            Set `RESUME=True`. The time budget may change. Checkpoints are written every
            100 batches and on a budget stop. Code 75 means a checkpointed pause, not success.
        """,
        ),
        cell(
            "code",
            "settings",
            """
            RUN_DIRECTORY = ''  # @param {type: "string"}
            RESUME = False  # @param {type: "boolean"}
            RUN_RESNET18 = True  # @param {type: "boolean"}
            RUN_SIGLIP = True  # @param {type: "boolean"}
            RESNET_EPOCHS = 3  # @param {type: "integer"}
            HEAD_EPOCHS = 30  # @param {type: "integer"}
            BATCH_SIZE = 64  # @param {type: "integer"}
            WORKERS = 2  # @param {type: "integer"}
            TIME_BUDGET_SECONDS = 1800  # @param {type: "integer"}
            SIGLIP_REVISION = 'main'  # @param {type: "string"}
            experiment = Path(RUN_DIRECTORY) if RUN_DIRECTORY else WORK / 'experiments' / uuid.uuid4().hex[:10]
            if RESUME and not experiment.is_dir(): raise ValueError('Resume requires an existing RUN_DIRECTORY.')
            if not RESUME: experiment.mkdir(parents=True,exist_ok=False)
            print('RUN_DIRECTORY:', experiment, '| save this path for resume')
            print('Model runtime:', MODEL_ENV['gpu'] or 'CPU')

            def training_command(*args, log_name):
                try:
                    run_holds(*args, log_name=log_name)
                    return True
                except RuntimeError as error:
                    if 'Command failed (75)' not in str(error): raise
                    print('Paused with checkpoint. Set RESUME=True and rerun with the same configuration.')
                    return False
        """,
        ),
        cell(
            "code",
            "train-resnet",
            """
            if RUN_RESNET18:
                resnet_run = experiment / 'resnet18'
                args = ['train-classifier','--architecture','resnet18','--dataset',CROP_DATA,
                        '--output',resnet_run,'--epochs',RESNET_EPOCHS,'--batch-size',BATCH_SIZE,
                        '--workers',WORKERS,'--time-budget-seconds',TIME_BUDGET_SECONDS]
                if RESUME and (resnet_run / 'last.pt').exists(): args += ['--resume']
                print('Fine-tuning ImageNet ResNet18…')
                training_command(*args,log_name='train_resnet18')
        """,
        ),
        cell(
            "code",
            "siglip-features",
            """
            if RUN_SIGLIP:
                FEATURE_CACHE = experiment / 'siglip_features'
                print('Extracting frozen SigLIP embeddings; completed batches are reused…')
                run_holds('extract-features','--dataset',CROP_DATA,'--output',FEATURE_CACHE,
                          '--batch-size',BATCH_SIZE,'--workers',WORKERS,'--revision',SIGLIP_REVISION,
                          log_name='siglip_features')
                print('Features ready:', FEATURE_CACHE)
        """,
        ),
        cell(
            "code",
            "siglip-head",
            """
            if RUN_SIGLIP:
                siglip_run = experiment / 'siglip'
                args = ['train-classifier','--architecture','siglip','--features',FEATURE_CACHE,
                        '--output',siglip_run,'--epochs',HEAD_EPOCHS,'--batch-size',512,
                        '--workers',0,'--device','cpu','--time-budget-seconds',TIME_BUDGET_SECONDS]
                if RESUME and (siglip_run / 'last.pt').exists(): args += ['--resume']
                print('Training a linear head on cached features (CPU)…')
                training_command(*args,log_name='train_siglip_head')
        """,
        ),
        cell(
            "markdown",
            "metrics-heading",
            """
            ## 3 · Results
            Best epochs are selected by **validation macro-F1**, never test performance.
            Reports contain accuracy, all-six-class macro-F1, per-class precision/recall/F1,
            confusion matrices and per-crop scores. These are crop metrics, not SAM detection quality.
            The Roboflow data overlaps the upstream YOLO/CNN training domain; evaluate on your gym separately.
        """,
        ),
        cell(
            "code",
            "metrics",
            """
            rows = []
            for architecture in ['resnet18','siglip']:
                folder = experiment / architecture
                if not (folder / 'metrics.json').exists():
                    print(architecture, 'not completed; resume its checkpoint if paused')
                    continue
                metrics = json.loads((folder / 'metrics.json').read_text())
                for split in ['validation','test']:
                    if split in metrics:
                        q = metrics[split]
                        rows.append({'Model':architecture,'Split':split,'Samples':q['sample_count'],
                                     'Accuracy':q['accuracy'],'Macro-F1':q['macro_f1_all_six_classes']})
                for filename in ['learning_curves.png','confusion_matrices.png']:
                    display(Image.open(folder / filename).copy())
            if rows: display(pd.DataFrame(rows).style.format(precision=3).hide(axis='index'))
        """,
        ),
        cell(
            "markdown",
            "inference-heading",
            """
            ## 4 · Use a trained classifier after SAM 3
            The artifact `resnet18/best.pt` or `siglip/best.pt` includes its encoder weights,
            class order and preprocessing. No Hugging Face access is needed to load the classifier.
            SAM 3 itself still needs its official checkpoint and CUDA.

            In the YOLO/SAM comparison notebook select `CLASSIFIER='trained'` and set
            `CLASSIFIER_WEIGHTS` to this artifact. Or use the CLI:

            ```bash
            python -m hold_detection.cli sam3 /path/to/photos --output /path/to/new-run \\
              --classifier trained --classifier-weights /path/to/siglip/best.pt
            ```

            Training uses GT boxes; inference uses imperfect SAM boxes. Validate this domain shift.
            Neither classifier has a trained `not_hold` class yet. Low scores become `unknown`,
            but high-confidence background errors remain possible.
        """,
        ),
        cell(
            "code",
            "export",
            """
            import shutil
            (experiment / 'session.json').write_text(json.dumps({'source_commit':CODE_SHA,
                'runtime':runtime_info(PYTHON,PROJECT),'crop_dataset':str(CROP_DATA)},indent=2))
            EXPORT_FEATURE_CACHE = False  # @param {type: "boolean"}
            from zipfile import ZipFile, ZIP_DEFLATED
            archive_path = WORK / ('classifiers_' + uuid.uuid4().hex[:8] + '.zip')
            with ZipFile(archive_path,'x',compression=ZIP_DEFLATED) as archive:
                for path in sorted(experiment.rglob('*')):
                    if path.is_symlink(): raise ValueError('Symlinks are not exported')
                    if not EXPORT_FEATURE_CACHE and 'siglip_features' in path.relative_to(experiment).parts: continue
                    if path.is_file(): archive.write(path,path.relative_to(experiment).as_posix())
            print('Download models/results before ending the session. Feature cache excluded by default.')
            files.download(str(archive_path))
        """,
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"name": "DeMoSearch · Classifier training", "provenance": []},
            "demosearch": {
                "training_version": 1,
                "full_dataset_training_validated": False,
            },
        },
        "cells": cells,
    }


if __name__ == "__main__":
    OUTPUT.write_text(
        json.dumps(build_training_notebook(), ensure_ascii=False, indent=1) + "\n"
    )
    print(OUTPUT)
