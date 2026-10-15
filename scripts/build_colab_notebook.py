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
            **Один набор фото → две модели → наглядное сравнение → измеримые метрики.**

            | Эксперимент | Локализация | Тип зацепа |
            |:---|:---|:---|
            | **YOLO11s baseline** | Bounding boxes | Шесть классов из detector head |
            | **SAM 3 + CNN** | Text-prompted instance masks | Отдельная crop CNN |

            Начни с 3 кадров. Включи **Runtime → Change runtime type → GPU**, Python 3.12.
            YOLO работает и на CPU, без Hugging Face. Для SAM нужны GPU и одобренный
            доступ к [facebook/sam3](https://huggingface.co/facebook/sam3).

            **Результаты:** оригинал / YOLO / SAM рядом, выбор кадра и порогов,
            сетка crops, latency и score plots. С COCO-разметкой — precision/recall/F1,
            box AP, mask AP/IoU и confusion matrix типов. **Без GT качество не выдумывается.**
        """),
        cell("markdown", "setup-heading", """
            ## 1 · Окружение
            Служебные ячейки свёрнуты. Используется готовый CUDA PyTorch Colab;
            зависимости устанавливаются в отдельный venv. Код берётся из нашего
            публичного GitHub, без base64-блоков; точный commit сохраняется в результатах.
        """),
        cell("code", "bootstrap", """
            # @title Подготовить окружение (один раз)
            import json, os, subprocess, sys, uuid
            from pathlib import Path
            from datetime import datetime, timezone
            import torch

            CODE_REF = 'main'  # @param {type: "string"}
            WORK = Path('/content/demosearch-comparison')
            PROJECT = WORK / 'project'
            WORK.mkdir(parents=True, exist_ok=True)
            if sys.version_info[:2] != (3, 12):
                raise RuntimeError('Выбери Python 3.12 runtime: зависимости SAM используют NumPy 1.26.')
            repository = 'https://github.com/DiasKzStrong/hold_detection.git'
            if not PROJECT.exists():
                subprocess.run(['git', 'clone', '--quiet', '--depth', '1', repository, str(PROJECT)], check=True)
            if not (PROJECT / '.git').is_dir():
                raise RuntimeError('Папка project занята другими файлами; выбери другой WORK.')
            dirty = subprocess.check_output(['git', '-C', str(PROJECT), 'status', '--porcelain'], text=True)
            if dirty.strip():
                raise RuntimeError('Сохрани изменения в project перед обновлением; они не перезаписываются.')
            subprocess.run(['git', '-C', str(PROJECT), 'fetch', '--quiet', 'origin', CODE_REF], check=True)
            CODE_SHA = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'FETCH_HEAD'], text=True).strip()
            previous = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
            if previous != CODE_SHA and 'hold_detection' in sys.modules:
                raise RuntimeError('Код обновился: перезапусти runtime, чтобы не смешивать версии модулей.')
            subprocess.run(['git', '-C', str(PROJECT), 'checkout', '--quiet', '--detach', CODE_SHA], check=True)
            if str(PROJECT / 'src') not in sys.path: sys.path.insert(0, str(PROJECT / 'src'))
            from hold_detection.notebook_runtime import prepare_runtime, run_cli, runtime_info
            PYTHON = prepare_runtime(WORK, PROJECT)

            def run_holds(*arguments, log_name='command'):
                return run_cli(PYTHON, PROJECT, list(arguments), WORK / 'logs' / (log_name + '.log'))

            print('Ready ·', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU baseline')
            print('Code commit:', CODE_SHA, '| installation logs:', WORK / 'logs')
        """, hidden=True),
        cell("markdown", "data-heading", """
            ## 2 · Данные и параметры
            Загрузи JPEG/PNG, ZIP с фото или `wallA_1_raw_data.zip`. Из Polycam выбираются
            RGB-кадры, а не depth PNG. Для больших наборов есть режим Google Drive.
        """),
        cell("code", "upload-inputs", """
            # @title Загрузить фото
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
                    raise ValueError('Выбери фото или один ZIP с простыми именами файлов.')
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

            if MAX_IMAGES < 0 or ROTATE not in (-1, 0, 90, 180, 270):
                raise ValueError('MAX_IMAGES >= 0; ROTATE = -1, 0, 90, 180 или 270')
            if not all(0 <= v <= 1 for v in [CANDIDATE_FLOOR, YOLO_THRESHOLD, SAM_THRESHOLD, MIN_TYPE_SCORE]):
                raise ValueError('Scores должны быть в [0,1]')
            if CANDIDATE_FLOOR > min(YOLO_THRESHOLD, SAM_THRESHOLD):
                raise ValueError('Candidate floor должен быть не выше порогов просмотра.')
            rotation = (90 if prepared.kind == 'polycam' else 0) if ROTATE == -1 else ROTATE
            experiment_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
            EXPERIMENT = PROJECT / 'runs' / ('comparison_' + experiment_id)
            EXPERIMENT.mkdir(parents=True, exist_ok=False)
            RUNS, THRESHOLDS = {}, {}
            print('Preview rotation:', rotation, '| images:', MAX_IMAGES or 'all')
        """),
        cell("markdown", "protocol-note", """
            `MAX_IMAGES=0` — все кадры. `ROTATE=-1` — 90° для нашего Polycam-скана,
            иначе 0°; проверь preview. Низкий `CANDIDATE_FLOOR` сохраняет proposals
            для AP, а пороги просмотра скрывают слабые предсказания без повторного inference.
            AP ограничен этим floor. Scores YOLO и SAM **не откалиброваны друг относительно друга**.
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
            Детекция и тип из одной модели. Этот блок не требует HF_TOKEN.
            Время модели измеряется отдельно от экспорта картинок; загрузка весов исключена,
            первый кадр включает warmup. На GPU таймеры синхронизированы.
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
            ## 4 · SAM 3 + crop CNN
            SAM сам предлагает masks по тексту; YOLO boxes ему не передаются.
            Локализация сравнивается отдельно от качества типов.
            При `RUN_SAM3=False` или отсутствии GPU остаётся полноценный YOLO baseline.
        """),
        cell("code", "sam-setup", """
            # @title Подготовить SAM 3 и доступ к весам
            import getpass
            SAM_READY = False
            if RUN_SAM3 and torch.cuda.is_available():
                print('Preparing SAM 3; installation and checkpoint download may take several minutes…')
                from google.colab import userdata
                try:
                    if not os.environ.get('HF_TOKEN'):
                        try:
                            token = userdata.get('HF_TOKEN')
                        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                            token = getpass.getpass('Hugging Face read token (hidden): ').strip()
                        if not token: raise ValueError('Нужен токен аккаунта с одобренным доступом к facebook/sam3.')
                        os.environ['HF_TOKEN'] = token
                        del token
                    PYTHON = prepare_runtime(WORK, PROJECT, sam3=True)
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
            RUNS.pop('SAM 3 + CNN', None)
            THRESHOLDS.pop('SAM 3 + CNN', None)
            if SAM_READY:
                sam_path = EXPERIMENT / ('sam3_' + uuid.uuid4().hex[:6])
                args = ['sam3', SOURCE, '--output', sam_path, '--rotate', rotation,
                        '--prompt', PROMPT, '--conf', CANDIDATE_FLOOR, '--max-det', 500,
                        '--classifier', 'cnn', '--min-type-score', MIN_TYPE_SCORE]
                if MAX_IMAGES: args += ['--limit', MAX_IMAGES]
                print('Running SAM 3 segmentation + CNN…')
                try:
                    run_holds(*args, log_name='sam3_inference')
                    RUNS['SAM 3 + CNN'] = sam_path
                    THRESHOLDS['SAM 3 + CNN'] = SAM_THRESHOLD
                    print('SAM 3 finished on the same frames as YOLO.')
                except RuntimeError as error:
                    print('SAM 3 inference failed:', error)
                    print('YOLO remains available. For GPU OOM, try a larger GPU and rerun this cell.')
        """),
        cell("markdown", "viewer-heading", """
            ## 5 · Сравнение кадр за кадром
            Выбери кадр, меняй пороги, включай masks/boxes. Подписи ID по умолчанию
            выключены, чтобы не закрывать зацепы. Типы и scores смотри в crops ниже.
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
            ## 6 · Ground truth — опционально
            Для quality metrics загрузи **COCO instance JSON** из CVAT/Roboflow.
            Разметь все видимые holds/volumes на оцениваемых кадрах; частичная разметка
            даст ложные FP. Названия images должны совпадать с файлами фото.

            Категории `hold` дают detection metrics; `crimp/jug/pinch/pocket/sloper/volume`
            дополнительно дают type metrics. Полигональные masks или COCO RLE нужны для mask AP.
            YOLO baseline не имеет masks: mask AP у него **N/A**, не 0.

            `source` — разметка исходных пикселей, поворот применяется автоматически.
            `processed` — разметка уже повёрнутых изображений. Без GT остаются только diagnostics.
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
                    raise ValueError('Выбери один COCO JSON.')
                GT_FILE = annotation_dir / next(iter(uploaded))
                del uploaded
            print('Quality metrics:', 'COCO annotations supplied' if GT_FILE else 'unavailable — no ground truth')
        """),
        cell("markdown", "metrics-heading", """
            ## 7 · Метрики
            Пороги берутся из viewer **на момент выполнения этой ячейки**. После изменения
            sliders выполни её снова. AP использует все сохранённые proposals, а P/R/F1 —
            выбранные пороги. Сопоставление class-agnostic, one-to-one, IoU ≥ 0.50, maxDets=500.
            AP50–95 имеет COCO IoU-сетку, но лимит 500 явно отличается от стандартных 100.

            Accuracy/F1 типов считаются на matched GT; отдельно показаны coverage и доля
            всех typed GT, которые найдены **и** классифицированы верно. Это не позволяет
            скрыть пропущенные holds за высокой conditional accuracy.
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
            ## 8 · Скачать эксперимент
            ZIP содержит обе галереи, masks/crops, comparison figures, metrics.csv,
            comparison.json и версии runtime. Токены и веса в архив не входят.
            Открой `report_…/index.html` после распаковки всей папки.
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
            **Ограничения эксперимента.** CNN — baseline: 69/100 correct на авторских
            sample-crops, не на нашем зале. Тип хвата может быть не виден с выбранного
            ракурса. Обе модели могут находить таблички вместо holds. Не дели соседние
            кадры одного скана случайно между train/test. Реальный SAM GPU inference
            в среде разработки не проверен; его выполняет этот Colab runtime.
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
