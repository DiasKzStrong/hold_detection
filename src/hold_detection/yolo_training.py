"""Validate labeled hold datasets and train/evaluate a six-class YOLO26 detector."""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

import yaml
from PIL import Image, ImageOps

HOLD_CLASSES = {'crimp', 'jug', 'pinch', 'pocket', 'sloper', 'volume'}


def download_roboflow_dataset(output, version, project='holdclassificatore_v3_mix'):
    """Download a versioned YOLO export using ROBOFLOW_API_KEY, without logging credentials."""
    import time
    from urllib.request import urlopen
    from urllib.parse import urlencode
    from urllib.error import HTTPError, URLError
    if project not in {'holdclassificatore_v3_mix', 'holdclassificatore_v2.2'} or version < 1:
        raise ValueError('Choose a supported HoldClassificatore project and a positive version number')
    key = os.environ.get('ROBOFLOW_API_KEY', '')
    if not key:
        raise ValueError('Set ROBOFLOW_API_KEY in the job environment; never put it in source code')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = (f'https://api.roboflow.com/climbing-holds-classification/{project}/{version}/yolov5pytorch?'
                + urlencode({'api_key': key}))
    temporary = None
    try:
        for attempt in range(120):
            with urlopen(endpoint, timeout=60) as response:
                payload = json.load(response)
            link = payload.get('export', {}).get('link')
            if link:
                break
            time.sleep(5)
        else:
            raise RuntimeError('Roboflow export is pending; retry later')
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
            temporary = Path(target.name)
            with urlopen(link, timeout=120) as response:
                shutil.copyfileobj(response, target)
        with ZipFile(temporary) as archive:
            if not any(name.endswith('data.yaml') for name in archive.namelist()):
                raise ValueError('Downloaded export is not a YOLO dataset')
        # Install without overwriting a concurrent/unknown download.
        os.link(temporary, output)
    except (HTTPError, URLError):
        raise RuntimeError('Roboflow download failed; check access/project/version. Credentials are hidden.') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    provenance = {'project': project, 'version': version, 'format': 'yolov5pytorch', 'path': str(output)}
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


def prepare_yolo_dataset(source, output):
    """Validate a Roboflow detection export and write portable paths, preserving class IDs."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError('Choose an output outside the source dataset.')
    with tempfile.TemporaryDirectory(prefix='holds-yolo-') as temporary:
        root = source
        if source.is_file():
            root = Path(temporary)
            with ZipFile(source) as archive:
                seen = set()
                if sum(item.file_size for item in archive.infolist()) > 20 * 1024**3:
                    raise ValueError('Dataset ZIP exceeds the 20 GiB extraction limit.')
                for item in archive.infolist():
                    path = PurePosixPath(item.filename)
                    if (path.is_absolute() or '..' in path.parts or '\\' in item.filename
                            or item.filename in seen or (item.external_attr >> 16) & 0o170000 == 0o120000):
                        raise ValueError(f'Unsafe ZIP entry: {item.filename}')
                    seen.add(item.filename)
                archive.extractall(root)
        configs = [p for p in root.rglob('data.yaml') if '__MACOSX' not in p.parts]
        if len(configs) != 1:
            raise ValueError('Expected one data.yaml in a YOLO detection export.')
        root = configs[0].parent.resolve()
        configuration = yaml.safe_load(configs[0].read_text())
        raw_names = configuration.get('names', [])
        if isinstance(raw_names, dict):
            indexed = {int(key): value for key, value in raw_names.items()}
            if sorted(indexed) != list(range(len(indexed))):
                raise ValueError('Class IDs must be contiguous from zero.')
            raw_names = [indexed[i] for i in range(len(indexed))]
        names = [{'slope': 'sloper'}.get(str(n).strip().lower(), str(n).strip().lower()) for n in raw_names]
        if len(names) != 6 or set(names) != HOLD_CLASSES:
            raise ValueError(f'Expected six HoldClassificatore classes, received {names}')

        files, splits, pixel_splits, families, warnings = [], {}, {}, {}, []
        fingerprint = hashlib.sha256(json.dumps(names).encode())
        for split, aliases in [('train', ['train']), ('val', ['valid', 'val', 'validation']), ('test', ['test'])]:
            folders = [root / alias for alias in aliases if (root / alias / 'images').is_dir()]
            folders += [root / 'images' / alias for alias in aliases if (root / 'images' / alias).is_dir()]
            if not folders and split == 'test':
                continue
            if len(folders) != 1:
                raise ValueError(f'Expected one {split} image folder; export YOLO detection format.')
            image_dir = folders[0] / 'images' if (folders[0] / 'images').is_dir() else folders[0]
            label_dir = (folders[0] / 'labels' if image_dir.name == 'images'
                         else root / 'labels' / image_dir.name)
            images = sorted(p for p in image_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if not images:
                raise ValueError(f'No images in {split}')
            counts = [0] * 6
            for image_path in images:
                relative = image_path.relative_to(image_dir)
                label_path = label_dir / relative.with_suffix('.txt')
                if not label_path.is_file():
                    raise ValueError(f'Missing label: {label_path}; use an empty TXT for a verified negative.')
                if not image_path.resolve().is_relative_to(root) or not label_path.resolve().is_relative_to(root):
                    raise ValueError('Dataset links must stay inside the export root.')
                with Image.open(image_path) as image:
                    if image.getexif().get(274, 1) != 1:
                        raise ValueError('Export auto-oriented images before training; EXIF rotation changes boxes.')
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                if pixel_hash in pixel_splits and pixel_splits[pixel_hash] != split:
                    raise ValueError(f'Identical image pixels cross dataset splits: {image_path.name}')
                pixel_splits[pixel_hash] = split
                family = image_path.stem.split('.rf.')[0]
                families.setdefault(family, set()).add(split)
                text = label_path.read_text()
                for number, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        raise ValueError(f'Expected class cx cy width height at {label_path}:{number}; masks are separate.')
                    values = [float(value) for value in parts]
                    label, cx, cy, width, height = values
                    if (not all(math.isfinite(v) for v in values) or not label.is_integer()
                            or not 0 <= label < 6 or not 0 < width <= 1 or not 0 < height <= 1
                            or cx - width / 2 < -1e-5 or cy - height / 2 < -1e-5
                            or cx + width / 2 > 1 + 1e-5 or cy + height / 2 > 1 + 1e-5):
                        raise ValueError(f'Invalid normalized box at {label_path}:{number}')
                    counts[int(label)] += 1
                fingerprint.update(f'{split}/{relative}:{pixel_hash}:{text}'.encode())
                files.append((split, relative, image_path, label_path))
            splits[split] = {'images': len(images), 'boxes': sum(counts), 'class_counts': counts}
        if any(count == 0 for count in splits['train']['class_counts']):
            warnings.append('Some classes have no training boxes; inspect class_counts before a full run.')
        warnings.append('Pixel hashes do not detect resized/mirrored images or neighboring views of the same wall.')
        output.mkdir(parents=True, exist_ok=False)
        for split, relative, image_path, label_path in files:
            for kind, original, target_name in [('images', image_path, relative), ('labels', label_path, relative.with_suffix('.txt'))]:
                target = output / split / kind / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
        # Keep dataset attribution and the original YAML without executing its download/path directives.
        provenance = output / 'provenance'
        provenance.mkdir()
        shutil.copy2(configs[0], provenance / 'source_data.yaml')
        for path in root.iterdir():
            if path.is_file() and path.name.lower().startswith(('readme', 'license')):
                shutil.copy2(path, provenance / path.name)
        # No fixed path: Ultralytics resolves these against this YAML's directory after moving to HPC.
        normalized = {'train': 'train/images', 'val': 'val/images', 'names': names, 'nc': 6}
        if 'test' in splits:
            normalized['test'] = 'test/images'
        data_yaml = output / 'data.yaml'
        data_yaml.write_text(yaml.safe_dump(normalized, sort_keys=False))
        report = {
            'schema_version': '1.0', 'source': str(source), 'names': names, 'splits': splits,
            'data_yaml': str(data_yaml), 'dataset_sha256': fingerprint.hexdigest(),
            'cross_split_filename_families': sorted(k for k, v in families.items() if len(v) > 1),
            'warnings': warnings,
        }
        (output / 'dataset_report.json').write_text(json.dumps(report, indent=2) + '\n')
        return report

class TrainingPaused(Exception):
    """Resume from the last completed epoch; an unfinished epoch is repeated."""


def train_yolo26(data, output, model='yolo26m.pt', epochs=100, imgsz=1024,
                 batch=8, device='0', workers=2, resume=False, stop_file=None):
    """Fine-tune the six-class detector and retain resumable completed-epoch checkpoints."""
    from importlib.metadata import version
    import signal
    import torch
    from ultralytics import YOLO

    if version('ultralytics') != '8.4.207':
        raise RuntimeError('Use ultralytics==8.4.207 from requirements-yolo26.txt.')
    data, output = Path(data).resolve(), Path(output).resolve()
    report_path = data.parent / 'dataset_report.json'
    if not data.is_file() or not report_path.is_file():
        raise ValueError('Run prepare_yolo_dataset first, then pass its data.yaml.')
    report = json.loads(report_path.read_text())
    if (output / 'TRAINING_COMPLETE').is_file():
        raise ValueError('This run already completed. Evaluate it or choose a new output directory.')
    if epochs < 1 or imgsz < 32 or batch < 1 or workers < 0:
        raise ValueError('Epochs/batch must be positive, imgsz >= 32, workers >= 0.')
    rank = int(os.environ.get('RANK', '-1'))
    leader = rank in {-1, 0}
    checkpoint = output / 'weights/last.pt'
    manifest = output / 'training_manifest.json'
    if resume:
        if not checkpoint.is_file() or not manifest.is_file():
            raise ValueError('Resume requires weights/last.pt and training_manifest.json.')
        previous = json.loads(manifest.read_text())
        if previous['dataset_sha256'] != report['dataset_sha256']:
            raise ValueError('Resume dataset differs from the original training dataset.')
        if previous['epochs'] != epochs or previous['imgsz'] != imgsz or previous['model'] != model:
            raise ValueError('Resume the same model, epoch budget and image size; use a new run for a new experiment.')
    elif leader and output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new run directory or resume: {output}')

    output.mkdir(parents=True, exist_ok=True)
    stop_file = Path(stop_file).resolve() if stop_file else output / 'STOP_REQUESTED'
    paused = output / 'PAUSED'
    if resume and leader and stop_file.exists():
        stop_file.unlink()
    if leader and paused.exists():
        paused.unlink()

    def request_pause(signum, frame):
        stop_file.touch()

    detector = YOLO(str(checkpoint) if resume else str(model), task='detect')

    def record_run(trainer):
        if leader:
            initial_weights = Path(str(getattr(detector, 'ckpt_path', '')))
            manifest.write_text(json.dumps({
                'dataset_sha256': report['dataset_sha256'], 'names': report['names'],
                'ultralytics': version('ultralytics'), 'torch': torch.__version__,
                'model': model, 'epochs': epochs, 'imgsz': imgsz, 'batch': batch,
                'device': device, 'seed': 42, 'resume': resume,
                'initial_weights_sha256': (previous.get('initial_weights_sha256') if resume else
                                           hashlib.sha256(initial_weights.read_bytes()).hexdigest()
                                           if initial_weights.is_file() else None),
                'resume_checkpoint_sha256': (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                             if resume else None),
            }, indent=2) + '\n')

    def pause_after_checkpoint(trainer):
        # All torchrun ranks enter this callback. Broadcast rank 0's decision to stop together.
        flag = torch.tensor(int(stop_file.exists() and checkpoint.is_file()), device=trainer.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(flag, src=0)
        if flag.item():
            if leader:
                paused.touch()
            raise TrainingPaused('Paused; resume from the last completed epoch.')

    detector.add_callback('on_train_start', record_run)
    detector.add_callback('on_train_batch_end', pause_after_checkpoint)
    arguments = dict(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                     device=device, workers=workers, project=str(output.parent), name=output.name,
                     exist_ok=True, seed=42, deterministic=True, cache=False, amp=True,
                     save=True, save_period=1, patience=25, max_det=500, plots=True,
                     close_mosaic=10, fliplr=0.5, flipud=0.0, degrees=5.0)
    try:
        previous_handlers = {}
        for sig in {signal.SIGTERM, getattr(signal, 'SIGUSR1', signal.SIGTERM)}:
            previous_handlers[sig] = signal.signal(sig, request_pause)
        if resume:
            arguments['resume'] = True
        detector.train(**arguments)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if leader:
        (output / 'TRAINING_COMPLETE').touch()
    return str(output / 'weights/best.pt')


def evaluate_yolo26(data, output, split='val', device='0', imgsz=1024):
    """Evaluate validation-selected best weights; test is explicit and never selects the model."""
    from ultralytics import YOLO
    data, output = Path(data).resolve(), Path(output).resolve()
    if split not in {'val', 'test'}:
        raise ValueError('Evaluation split must be val or test.')
    if split not in yaml.safe_load(data.read_text()):
        raise ValueError(f'No {split} split in this dataset.')
    best = output / 'weights/best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    model = YOLO(str(best), task='detect')
    metrics = model.val(data=str(data), split=split, device=device, imgsz=imgsz,
                        max_det=500, workers=0, project=str(output), name=f'evaluation_{split}',
                        exist_ok=True, plots=True)
    result = {'split': split, 'metrics': metrics.results_dict,
              'names': model.names, 'per_class_map50_95': metrics.box.maps.tolist(),
              'weights_sha256': hashlib.sha256(best.read_bytes()).hexdigest()}
    (output / f'metrics_{split}.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Notebook-first YOLO26 hold-type training')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('--output', required=True, type=Path)
    train = commands.add_parser('train')
    train.add_argument('--data', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path)
    train.add_argument('--model', default='yolo26m.pt')
    train.add_argument('--epochs', type=int, default=100)
    train.add_argument('--imgsz', type=int, default=1024)
    train.add_argument('--batch', type=int, default=8)
    train.add_argument('--workers', type=int, default=2)
    train.add_argument('--device', default='0')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--stop-file', type=Path)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--data', required=True, type=Path)
    evaluate.add_argument('--output', required=True, type=Path)
    evaluate.add_argument('--split', choices=['val', 'test'], default='val')
    evaluate.add_argument('--device', default='0')
    evaluate.add_argument('--imgsz', type=int, default=1024)
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        if command == 'prepare':
            print(json.dumps(prepare_yolo_dataset(**args), indent=2))
        elif command == 'train':
            print(train_yolo26(**args))
        else:
            print(json.dumps(evaluate_yolo26(**args), indent=2))
    except TrainingPaused as error:
        print(error)
        raise SystemExit(75)


if __name__ == '__main__':
    main()
