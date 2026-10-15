"""Mechanically build a standalone notebook with the current pipeline embedded.

Regenerate after package edits: python scripts/build_colab_notebook.py
The embedded ZIP contains our source/docs only, never weights, photos or secrets.
"""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from textwrap import dedent
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "DeMoSearch_SAM3_Colab.ipynb"
SAM3_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"


def cell(kind, name, source):
    item = {"cell_type": kind, "id": name, "metadata": {},
            "source": dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def embedded_project():
    files = sorted((ROOT / "src" / "hold_detection").glob("*.py"))
    files += [ROOT / "docs" / "model-provenance.md"]
    buffer = BytesIO()
    manifest = {}
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as z:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            manifest[relative] = hashlib.sha256(content).hexdigest()
            item = ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = ZIP_DEFLATED
            z.writestr(item, content)
    payload = buffer.getvalue()
    return base64.b64encode(payload).decode(), hashlib.sha256(payload).hexdigest(), manifest


def build_notebook():
    payload, digest, manifest = embedded_project()
    cells = [
        cell("markdown", "intro", """
            # DeMoSearch: SAM 3 → hold masks → hold type

            Самостоятельный notebook для Google Colab. Код нашего пайплайна встроен
            в него; отдельное клонирование проекта не нужно.
            SAM 3 ищет и сегментирует зацепы по тексту. Отдельная CNN предсказывает
            `crimp / jug / pinch / pocket / sloper / volume`.

            **Перед запуском:**
            1. В Colab: **Runtime → Change runtime type → GPU**, Python 3.12.
            2. Получи доступ к [официальным весам SAM 3](https://huggingface.co/facebook/sam3).
               Аккаунт с токеном должен иметь одобренный доступ.
            3. Добавь read-token в **Colab Secrets** (значок ключа слева), имя `HF_TOKEN`,
               разреши доступ этому notebook. Альтернатива — скрытый ввод в ячейке авторизации.
            4. Выполняй ячейки сверху вниз. Начни с **одного фото**.

            Нужен NVIDIA GPU. При нехватке VRAM используй GPU с большей памятью.
            Фото и результаты по умолчанию остаются в текущей Colab-сессии; скачай
            итоговый ZIP перед её завершением. Drive подключается только по твоему выбору.

            В этой облачной машине нет GPU и заблокировано скачивание весов:
            здесь проверены структура notebook и вспомогательный код, но **полный
            SAM 3 inference на Colab ещё не выполнен**. CNN — baseline, а не
            надёжная разметка: на 100 авторских sample-crops получено 69% accuracy.
        """),
        cell("code", "gpu-check", """
            import sys
            import torch
            import torchvision
            from packaging.version import Version

            if sys.version_info[:2] != (3, 12):
                raise RuntimeError("Нужен runtime с Python 3.12: NumPy 1.26 не поддерживает Python 3.13+.")
            if Version(torch.__version__.split('+')[0]) < Version('2.7'):
                raise RuntimeError("Нужен PyTorch >= 2.7. Выбери свежий Colab GPU runtime.")
            if not torch.cuda.is_available():
                raise RuntimeError("Включи GPU: Runtime → Change runtime type → GPU, затем повтори.")
            props = torch.cuda.get_device_properties(0)
            print('Python:', sys.version.split()[0])
            print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
            print('GPU:', props.name, '| VRAM GiB:', round(props.total_memory / 1024**3, 1))
            print('CUDA:', torch.version.cuda)
        """),
        cell("markdown", "setup-info", """
            ## Установка

            Создаётся отдельный venv с доступом к **готовому CUDA PyTorch из Colab**.
            NumPy < 2 и остальные зависимости устанавливаются внутри него; пакеты
            kernel Colab не заменяются. SAM 3 фиксируется на конкретном Git commit.
            Первый запуск может занять несколько минут. Встроенный ZIP ниже содержит
            только исходники нашего пайплайна и описание checkpoint provenance.
        """),
    ]
    bootstrap = '''
import base64
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

WORK = Path('/content/demosearch')
PROJECT = WORK / 'project'
WORK.mkdir(parents=True, exist_ok=True)
PROJECT.mkdir(parents=True, exist_ok=True)
PAYLOAD_B64 = __PAYLOAD__
PAYLOAD_SHA256 = __DIGEST__
payload = base64.b64decode(PAYLOAD_B64, validate=True)
if hashlib.sha256(payload).hexdigest() != PAYLOAD_SHA256:
    raise RuntimeError('Embedded project checksum mismatch')
with zipfile.ZipFile(BytesIO(payload)) as archive:
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        if relative.is_absolute() or '..' in relative.parts or '\\\\' in member.filename:
            raise ValueError('Unsafe embedded path')
        target = PROJECT / member.filename
        content = archive.read(member)
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f'{target}: сохрани свои изменения или создай новый runtime')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
print('Embedded pipeline extracted:', PROJECT)
'''
    bootstrap = bootstrap.replace("__PAYLOAD__", repr(payload)).replace("__DIGEST__", repr(digest))
    cells.append(cell("code", "bootstrap", bootstrap))
    cells.append(cell("code", "install", f'''
        import importlib.util
        import venv

        VENV = WORK / 'venv'
        if not (VENV / 'bin' / 'python').exists():
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(VENV)
        PYTHON = VENV / 'bin' / 'python'
        requirements = [
            'numpy==1.26.4', 'opencv-python==4.11.0.86', 'setuptools>=77,<81',
            'wheel>=0.43,<1', 'einops==0.8.1', 'pycocotools==2.0.11',
            'decord==0.6.0', 'psutil>=5.9', 'pillow>=11,<13',
            'huggingface-hub==2.2.0', 'timm==1.0.30', 'iopath==0.1.10',
            'ftfy==6.1.1', 'regex', 'requests',
        ]
        # Reuse the Triton version paired with Colab's CUDA PyTorch.
        if importlib.util.find_spec('triton') is None:
            raise RuntimeError('Triton отсутствует в CUDA runtime. Выбери свежий Colab GPU runtime.')
        subprocess.run([str(PYTHON), '-m', 'pip', 'install', *requirements], check=True)
        subprocess.run([
            str(PYTHON), '-m', 'pip', 'install', '--no-build-isolation',
            'git+https://github.com/facebookresearch/sam3.git@{SAM3_REVISION}',
        ], check=True)

        def command_env():
            environment = os.environ.copy()
            environment['PYTHONPATH'] = str(PROJECT / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
            environment['HF_HUB_DISABLE_XET'] = '1'
            return environment

        def run_holds(*arguments):
            return subprocess.run(
                [str(PYTHON), '-m', 'hold_detection.cli', *map(str, arguments)],
                cwd=PROJECT, env=command_env(), check=True,
            )

        subprocess.run([
            str(PYTHON), '-c',
            "import torch; from sam3.model_builder import build_sam3_image_model; "
            "from sam3.model.sam3_image_processor import Sam3Processor; "
            "assert torch.cuda.is_available(), 'CUDA недоступна в дочернем runtime'; "
            "print('SAM 3 imports OK; CUDA GPU:', torch.cuda.get_device_name(0))",
        ], cwd=PROJECT, env=command_env(), check=True)
    '''))
    cells.append(cell("markdown", "auth-info", """
        ## Доступ к весам

        Ячейка сначала проверяет `HF_TOKEN` в окружении и Colab Secrets, затем
        предлагает скрытый ввод. Токен не выводится и не записывается в notebook
        или файлы авторизации. Одного токена недостаточно без одобрения доступа
        к `facebook/sam3`. При HTTP 401/403 сначала проверь это одобрение и права токена.
    """))
    cells.append(cell("code", "authentication", """
        import getpass
        from google.colab import userdata

        token = os.environ.get('HF_TOKEN')
        if not token:
            try:
                token = userdata.get('HF_TOKEN')
            except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                token = None
        if not token:
            token = getpass.getpass('Hugging Face read token (скрытый ввод): ').strip()
        if not token:
            raise RuntimeError('Нужен HF_TOKEN аккаунта с доступом к facebook/sam3.')
        os.environ['HF_TOKEN'] = token
        del token
        print('Токен передан текущей сессии; его значение не выводится.')
    """))
    cells.append(cell("code", "download-weights", """
        # Several GB: official SAM 3 weights. Existing downloads are reused.
        run_holds('fetch-sam3')
        # Small, checksum-verified reference CNN for optional type classification.
        run_holds('fetch-models', '--model', 'classifier')
        run_holds('sam3-doctor')
    """))
    cells.append(cell("markdown", "input-info", """
        ## Фото или Polycam ZIP

        `upload`: выбери JPEG/PNG, несколько фото, ZIP с фото или
        **`wallA_1_raw_data.zip`**. Не загружай OBJ/GLB: это геометрия/текстуры,
        а для поиска зацепов нужны RGB-кадры.

        Для больших данных выбери `drive` и укажи путь к ZIP или папке фотографий.
        Подключение Drive будет запрошено только в этом режиме. Из Polycam автоматически
        выбираются `corrected_images`, а не depth/confidence PNG.
    """))
    cells.append(cell("code", "upload-inputs", """
        from datetime import datetime, timezone
        import uuid
        from google.colab import files, drive
        from hold_detection.colab_support import prepare_inputs

        INPUT_MODE = 'upload'  # @param ["upload", "drive"]
        DRIVE_INPUT = '/content/drive/MyDrive/DeMoSearch/wallA_1_raw_data.zip'  # @param {type: "string"}

        session_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        if INPUT_MODE == 'drive':
            drive.mount('/content/drive')
            selected_paths = [Path(DRIVE_INPUT)]
        elif INPUT_MODE == 'upload':
            upload_dir = WORK / 'uploads' / session_id
            upload_dir.mkdir(parents=True, exist_ok=False)
            old_directory = Path.cwd()
            try:
                os.chdir(upload_dir)
                uploaded = files.upload()
            finally:
                os.chdir(old_directory)
            if not uploaded:
                raise RuntimeError('Файлы не выбраны; повтори ячейку.')
            if any(Path(name).name != name for name in uploaded):
                raise ValueError('Ожидались простые имена загруженных файлов.')
            selected_paths = [upload_dir / name for name in uploaded]
            del uploaded
        else:
            raise ValueError('INPUT_MODE должен быть upload или drive')
        prepared = prepare_inputs(selected_paths, WORK / 'prepared' / session_id)
        SOURCE = prepared.source
        print('Источник:', prepared.kind, '| RGB-фото:', prepared.image_count)
        print('Папка кадров:', SOURCE)
    """))
    cells.append(cell("markdown", "settings-info", """
        ## Параметры и предпросмотр

        `ROTATE=-1` выбирает 90° для нашего Polycam-скана и 0° для обычных фото.
        Проверь предпросмотр: для другого скана явно укажи 0/90/180/270.
        `MAX_IMAGES=1` — быстрый первый тест; **0 = обработать все фото**.
        `CLASSIFY=True` добавляет CNN после SAM. При False сохраняются маски,
        но типы остаются `unknown`. SAM confidence не является уверенностью в типе хвата.
    """))
    cells.append(cell("code", "settings-preview", """
        from PIL import Image
        from IPython.display import display
        from hold_detection.pipeline import collect_images

        PROMPT = 'climbing hold'  # @param {type: "string"}
        SAM_CONFIDENCE = 0.5  # @param {type: "number"}
        MIN_TYPE_SCORE = 0.5  # @param {type: "number"}
        MAX_IMAGES = 1  # @param {type: "integer"}
        ROTATE = -1  # @param {type: "integer"}
        CLASSIFY = True  # @param {type: "boolean"}

        if ROTATE not in (-1, 0, 90, 180, 270):
            raise ValueError('ROTATE: -1 (auto), 0, 90, 180 или 270')
        if not 0 <= SAM_CONFIDENCE <= 1 or not 0 <= MIN_TYPE_SCORE <= 1 or MAX_IMAGES < 0:
            raise ValueError('Scores должны быть в [0,1], MAX_IMAGES >= 0')
        if not PROMPT.strip():
            raise ValueError('PROMPT не может быть пустым')
        rotation = (90 if prepared.kind == 'polycam' else 0) if ROTATE == -1 else ROTATE
        image_paths = collect_images(SOURCE)
        with Image.open(image_paths[0]) as image:
            preview = image.convert('RGB').rotate(-rotation, expand=True)
        preview.thumbnail((800, 800))
        print('Предпросмотр:', image_paths[0].name, '| поворот по часовой:', rotation)
        display(preview)
    """))
    cells.append(cell("code", "run-inference", """
        # A fresh output directory is created for every run; previous results survive.
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
        RUN_DIR = PROJECT / 'runs' / ('sam3_' + run_id)
        arguments = [
            'sam3', SOURCE, '--output', RUN_DIR,
            '--prompt', PROMPT, '--conf', SAM_CONFIDENCE,
            '--min-type-score', MIN_TYPE_SCORE, '--rotate', rotation,
            '--classifier', 'cnn' if CLASSIFY else 'none',
        ]
        if MAX_IMAGES:
            arguments.extend(['--limit', MAX_IMAGES])
        run_holds(*arguments)

        # Record only software/hardware metadata, never environment variables or tokens.
        runtime_probe = (
            "import json, sys, torch, numpy; from importlib.metadata import version; "
            "print(json.dumps({'python':sys.version.split()[0], 'torch':torch.__version__, "
            "'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0), "
            "'numpy':numpy.__version__, 'sam3':version('sam3'), "
            "'torchvision':version('torchvision'), 'triton':version('triton')}))"
        )
        probe = subprocess.run([str(PYTHON), '-c', runtime_probe], env=command_env(),
                               check=True, text=True, capture_output=True)
        (RUN_DIR / 'colab_runtime.json').write_text(probe.stdout, encoding='utf-8')
        summary = json.loads((RUN_DIR / 'summary.json').read_text())
        print('Готово:', summary['frame_count'], 'кадров,', summary['detection_count'], 'наблюдений зацепов')
        print('Это не число уникальных/правильно найденных зацепов; нужны ground-truth labels.')
    """))
    cells.append(cell("markdown", "results-info", """
        ## Посмотреть результат прямо здесь

        Цветные маски и подписи ниже — результат текущего запуска. Затем показаны
        первые crops и таблица scores. Высокий score не гарантирует правильный тип:
        рабочая поверхность может быть скрыта, а модель может ошибаться.
    """))
    cells.append(cell("code", "inline-results", """
        from html import escape
        from IPython.display import HTML

        SHOW_FRAMES = 3
        SHOW_CROPS = 12
        for frame_path in sorted(RUN_DIR.glob('*/holds.json'))[:SHOW_FRAMES]:
            frame = json.loads(frame_path.read_text())
            print(Path(frame['source_image']).name, '| candidates:', len(frame['holds']))
            with Image.open(frame_path.parent / 'annotated.jpg') as annotation:
                display(annotation.copy())
            rows = []
            for hold in frame['holds']:
                label = hold['grip_type'] or hold['object_kind']
                score = '—' if hold['type_score'] is None else f"{hold['type_score']:.3f}"
                rows.append('<tr>' + ''.join('<td>' + escape(str(value)) + '</td>' for value in
                    [hold['hold_id'], label, score, round(hold['detector_score'], 3),
                     ', '.join(hold['review_reasons'])]) + '</tr>')
            display(HTML('<table><tr><th>ID</th><th>Type</th><th>Type score</th>'
                         '<th>SAM score</th><th>Review</th></tr>' + ''.join(rows) + '</table>'))
            for hold in frame['holds'][:SHOW_CROPS]:
                print(hold['hold_id'], '|', hold['grip_type'] or hold['object_kind'])
                with Image.open(frame_path.parent / hold['crop_path']) as crop:
                    thumbnail = crop.copy()
                thumbnail.thumbnail((220, 180))
                display(thumbnail)
    """))
    cells.append(cell("code", "download-results", """
        from hold_detection.colab_support import bundle_results

        RESULT_ZIP = bundle_results(RUN_DIR, WORK / (RUN_DIR.name + '_' + uuid.uuid4().hex[:6] + '_results.zip'))
        print('Скачай ZIP, распакуй всю папку и открой index.html в браузере.')
        print('Внутри: annotated.jpg, crops/, masks/*.png, holds.json, summary.json и версии runtime.')
        files.download(str(RESULT_ZIP))
    """))
    cells.append(cell("markdown", "troubleshooting", """
        ## Если что-то не получилось

        - **Нет GPU / CUDA:** включи GPU и переподключись к runtime; повтори ячейки сверху.
        - **Hugging Face 401/403:** проверь одобренный доступ к `facebook/sam3` именно
          у владельца токена, read permissions и доступ notebook к Colab Secret.
        - **CUDA out of memory:** оставь одно фото, используй GPU с большей памятью.
          Само ограничение числа фото не уменьшает память обработки одного кадра.
        - **Слишком мало зацепов:** сравни prompts `climbing hold` / `rock climbing hold`,
          попробуй confidence 0.3; это также может увеличить число ложных объектов.
        - **Неверные типы:** это известное ограничение baseline CNN. Проверь crops и
          собери собственную разметку; маски и type scores оцениваются отдельно.
        - **Кадр повёрнут:** исправь ROTATE и заново выполни параметры и inference.

        Повторный inference и повторная упаковка создают новые файлы без перезаписи старых.
        Скачивай всю папку результатов: один index.html без изображений не сработает.
    """))
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU", "colab": {"name": OUTPUT.name, "provenance": []},
            "demosearch": {"sam3_revision": SAM3_REVISION, "embedded_files": manifest,
                           "gpu_inference_validated": False},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
