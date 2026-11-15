"""Audit VisDrone annotations and write reproducible dataset statistics.
Run: python -m src.analyze_data
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from .common import ROOT, SPLITS, config, dataset_root


def parse_annotation(path: Path):
    with path.open(encoding='utf-8-sig') as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            fields = [s.strip() for s in line.split(',')]
            if len(fields) != 8:
                yield line_number, None, 'expected 8 fields'
                continue
            try:
                values = [int(v) for v in fields]
            except ValueError:
                yield line_number, None, 'non-integer field'
                continue
            yield line_number, values, None


def inspect(root: Path, cfg: dict, check_hashes=False):
    objects, images, errors = [], [], []
    signatures = {}
    split_names = SPLITS
    for split, folder in split_names.items():
        image_dir, ann_dir = root / folder / 'images', root / folder / 'annotations'
        if not image_dir.is_dir() or not ann_dir.is_dir():
            errors.append({'split': split, 'file': str(root / folder), 'error': 'missing images or annotations directory'})
            continue
        paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        ann_files = set(ann_dir.glob('*.txt'))
        image_stems = {p.stem for p in paths}
        for ann in sorted(ann_files):
            if ann.stem not in image_stems:
                errors.append({'split': split, 'file': str(ann), 'error': 'orphan annotation'})
        for img in tqdm(paths, desc=f'Auditing {split}'):
            ann = ann_dir / f'{img.stem}.txt'
            if not ann.exists():
                errors.append({'split': split, 'file': str(img), 'error': 'missing annotation'})
                continue
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    w, h = im.size
                if w <= 0 or h <= 0:
                    raise ValueError('invalid dimensions')
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                errors.append({'split': split, 'file': str(img), 'error': f'invalid image: {exc}'})
                continue
            if check_hashes:
                digest = hashlib.sha256(img.read_bytes()).hexdigest()
                if digest in signatures and signatures[digest] != split:
                    errors.append({'split': split, 'file': str(img), 'error': f'exact image duplicate across {signatures[digest]} and {split}'})
                signatures.setdefault(digest, split)
            valid_count = 0
            ignored_count = 0
            for line, v, problem in parse_annotation(ann):
                if problem:
                    errors.append({'split': split, 'file': str(ann), 'line': line, 'error': problem})
                    continue
                x, y, bw, bh, score, category, truncation, occlusion = v
                if score not in (0, 1) or category not in range(12) or truncation not in (0, 1) or occlusion not in (0, 1, 2, 3):
                    errors.append({'split': split, 'file': str(ann), 'line': line, 'error': 'invalid attribute/category'})
                    continue
                if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
                    errors.append({'split': split, 'file': str(ann), 'line': line, 'error': 'invalid/out-of-image bounding box'})
                    continue
                if score == 0 or category not in range(1, 11):
                    ignored_count += 1
                    continue
                valid_count += 1
                area = bw * bh
                size = 'small' if area < cfg['small_area'] else ('medium' if area < cfg['medium_area'] else 'large')
                objects.append({'split': split, 'image': img.name, 'class_id': category - 1,
                                'class_name': cfg['classes'][category - 1], 'width': bw, 'height': bh,
                                'area': area, 'relative_area': area / (w * h), 'size': size,
                                'occlusion': occlusion, 'truncation': truncation})
            images.append({'split': split, 'image': img.name, 'width': w, 'height': h,
                           'valid_objects': valid_count, 'ignored_objects': ignored_count})
        expected = cfg['expected_images'][split]
        if len(paths) != expected:
            errors.append({'split': split, 'file': str(image_dir), 'error': f'expected {expected} images, found {len(paths)}'})
    return pd.DataFrame(images), pd.DataFrame(objects), pd.DataFrame(errors, columns=['split', 'file', 'line', 'error'])


def write_results(images, objects, errors, dest: Path, cfg: dict):
    dest.mkdir(parents=True, exist_ok=True)
    images.to_csv(dest / 'images.csv', index=False)
    objects.to_csv(dest / 'objects.csv', index=False)
    errors.to_csv(dest / 'validation_errors.csv', index=False)
    summary = {'image_count': images.groupby('split').size().to_dict() if not images.empty else {},
               'object_count': objects.groupby('split').size().to_dict() if not objects.empty else {},
               'errors': len(errors), 'size_definition': 'original-image bbox area in pixels: small <1024; medium <9216; large >=9216'}
    if not objects.empty:
        summary['class_counts'] = {split: {name: int((group['class_name'] == name).sum()) for name in cfg['classes']}
                                   for split, group in objects.groupby('split')}
        summary['size_counts'] = objects.groupby(['split', 'size']).size().rename('count').reset_index().to_dict('records')
        summary['occlusion_counts'] = objects.groupby(['split', 'occlusion']).size().rename('count').reset_index().to_dict('records')
        for name, column in [('class_distribution', 'class_name'), ('object_sizes', 'size'), ('occlusion_distribution', 'occlusion')]:
            counts = objects.groupby(['split', column]).size().unstack(fill_value=0)
            if name == 'class_distribution':
                counts = counts.reindex(columns=cfg['classes'], fill_value=0)
            counts.to_csv(dest / f'{name}.csv')
            ax = counts.T.plot(kind='bar', figsize=(12, 5))
            ax.set_ylabel('Number of objects')
            ax.set_title(name.replace('_', ' ').title())
            ax.figure.tight_layout()
            ax.figure.savefig(dest / f'{name}.png', dpi=160)
            plt.close(ax.figure)
        ax = objects['relative_area'].clip(lower=1e-8).plot.hist(bins=60, logx=True, figsize=(9, 5))
        ax.set_xlabel('Bounding box area / image area (log scale)')
        ax.figure.tight_layout()
        ax.figure.savefig(dest / 'relative_box_area.png', dpi=160)
        plt.close(ax.figure)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-hashes', action='store_true', help='Check exact duplicate image bytes across splits (slower)')
    args = parser.parse_args()
    cfg = config()
    images, objects, errors = inspect(dataset_root(), cfg, args.check_hashes)
    summary = write_results(images, objects, errors, ROOT / 'outputs' / 'data_analysis', cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Reports: {ROOT / "outputs/data_analysis"}')
    if len(errors):
        raise SystemExit(f'Dataset audit found {len(errors)} issues; see validation_errors.csv')

if __name__ == '__main__':
    main()
