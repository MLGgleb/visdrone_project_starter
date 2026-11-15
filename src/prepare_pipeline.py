"""Prepare immutable VisDrone data for YOLO and COCO consumers.
Run: python -m src.prepare_pipeline
"""
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image
import yaml
from .common import ROOT, SPLITS, config, dataset_root
from .analyze_data import parse_annotation


def annotations(path, width, height):
    """
    Чтение и проверка аннотаций VisDrone2019-DET.

    Обрабатывает:
    - пустые строки и файлы;
    - специальные категории 0 и 11;
    - рамки с нулевой площадью;
    - некорректные координаты.
    """
    result = []

    with open(path, "r", encoding="utf-8-sig") as file:
        for line_number, raw_line in enumerate(file, start=1):

            raw_line = raw_line.strip()

            # Пропускаем пустые строки
            if not raw_line:
                continue

            # Убираем завершающие запятые, если они есть
            raw_line = raw_line.rstrip(",")

            values = [v.strip() for v in raw_line.split(",")]

            if len(values) != 8:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"expected 8 fields, got {len(values)} "
                    f"({raw_line!r})"
                )

            try:
                x, y, w, h, score, cat, trunc, occ = map(
                    int, values
                )
            except ValueError:
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"non-integer annotation ({raw_line!r})"
                )

            # Проверка метаданных
            if not (
                0 <= cat <= 11
                and score in (0, 1)
                and trunc in (0, 1)
                and occ in (0, 1, 2, 3)
            ):
                raise ValueError(
                    f"{path}:{line_number}: invalid metadata "
                    f"(score={score}, category={cat}, "
                    f"truncation={trunc}, occlusion={occ})"
                )

            # Игнорируемые категории
            if score == 0 or cat in (0, 11):
                continue

            # Пропускаем рамки с нулевой площадью
            if w <= 0 or h <= 0:
                continue

            # Проверяем выход рамки за пределы изображения
            if (
                x < 0 or y < 0
                or x + w > width
                or y + h > height
            ):
                raise ValueError(
                    f"{path}:{line_number}: "
                    f"bounding box outside image "
                    f"(box={(x, y, w, h)}, "
                    f"image={(width, height)})"
                )

            result.append(
                (x, y, w, h, cat, trunc, occ)
            )

    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def link_or_copy(source, destination, method):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if method == 'hardlink':
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)


def prepare(source, output, method='hardlink', check_hashes=True):
    names = config()['classes']
    if output.resolve() == source.resolve():
        raise ValueError('Output cannot be the source directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'classes': names, 'splits': {}, 'source': str(source.resolve()),
                'ignored_regions_policy': 'exclude_from_targets; retain_original_annotations',
                'hash_algorithm': 'sha256' if check_hashes else None}
    seen = {}
    for split, folder in SPLITS.items():
        image_dir = source / folder / 'images'
        annotation_dir = source / folder / 'annotations'
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(f'Missing VisDrone split: {folder}')
        files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'})
        if not files:
            raise ValueError(f'No images in {image_dir}')
        coco = {'images': [], 'annotations': [], 'categories': [
            {'id': i+1, 'name': name, 'supercategory': 'object'} for i, name in enumerate(names)]}
        split_manifest = {'images': len(files), 'objects': 0}
        for image_id, image in enumerate(files, 1):
            annotation = annotation_dir / (image.stem + '.txt')
            if not annotation.exists():
                raise FileNotFoundError(annotation)
            with Image.open(image) as im:
                im.verify()
            with Image.open(image) as im:
                width, height = im.size
            boxes = annotations(annotation, width, height)
            if check_hashes:
                digest = sha256(image)
                if digest in seen and seen[digest][0] != split:
                    raise ValueError(f'Cross-split exact duplicate: {seen[digest][1]} and {image}')
                seen[digest] = (split, str(image))
            dest_image = output / 'images' / split / image.name
            link_or_copy(image, dest_image, method)
            yolo_lines = []
            for x, y, w, h, cat, trunc, occ in boxes:
                yolo_lines.append(f'{cat-1} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}')
                coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': image_id,
                   'category_id': cat, 'bbox': [x,y,w,h], 'area': w*h, 'iscrowd': 0,
                   'visdrone_truncation': trunc, 'visdrone_occlusion': occ})
            label = output / 'labels' / split / (image.stem + '.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text('\n'.join(yolo_lines) + ('\n' if yolo_lines else ''), encoding='utf-8')
            coco['images'].append({'id': image_id, 'file_name': image.name, 'width': width, 'height': height})
            split_manifest['objects'] += len(boxes)
        coco_path = output / 'annotations' / f'instances_{split}.json'
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_path.write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
        manifest['splits'][split] = split_manifest
    descriptor = {'path': str(output.resolve()), 'train': 'images/train', 'val': 'images/val',
                  'test': 'images/test', 'names': dict(enumerate(names))}
    (output / 'dataset.yaml').write_text(yaml.safe_dump(descriptor, allow_unicode=True, sort_keys=False), encoding='utf-8')
    config_bytes = (ROOT / 'configs' / 'data.yaml').read_bytes()
    manifest['config_sha256'] = hashlib.sha256(config_bytes).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=dataset_root())
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'prepared')
    parser.add_argument('--method', choices=['copy', 'hardlink'], default='hardlink')
    parser.add_argument('--no-hashes', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.method, not args.no_hashes), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
