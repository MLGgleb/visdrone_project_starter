"""Convert original VisDrone labels to YOLO labels, preserving original data."""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from .common import ROOT, SPLITS, dataset_root, config
from .analyze_data import parse_annotation


def convert(root: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for split, folder in SPLITS.items():
        source = root / folder
        if not source.exists():
            raise FileNotFoundError(source)
        dest = output / 'labels' / split
        dest.mkdir(parents=True, exist_ok=True)
        for image in tqdm(sorted((source / 'images').glob('*.jpg')), desc=f'YOLO {split}'):
            with Image.open(image) as im:
                w, h = im.size
            rows = []
            for line, v, err in parse_annotation(source / 'annotations' / f'{image.stem}.txt'):
                if err:
                    raise ValueError(f'{image.name}:{line}: {err}')
                x, y, bw, bh, score, cat, trunc, occ = v
                if score == 0 or not 1 <= cat <= 10:
                    continue
                if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w or y + bh > h:
                    raise ValueError(f'Invalid bbox in {image.name}:{line}')
                rows.append(f'{cat - 1} {(x + bw / 2) / w:.8f} {(y + bh / 2) / h:.8f} {bw / w:.8f} {bh / h:.8f}')
            (dest / f'{image.stem}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
    # Absolute image paths avoid symlinks and keep original dataset immutable.
    import yaml
    descriptor = {'path': str(root.resolve()),
                  'train': f'{SPLITS["train"]}/images',
                  'val': f'{SPLITS["val"]}/images',
                  'test': f'{SPLITS["test"]}/images',
                  'names': dict(enumerate(config()['classes']))}
    # YOLO expects labels alongside images. Use this conversion for analysis/export;
    # training needs a YOLO-compatible images/labels layout (planned in stage 4).
    (output / 'classes.yaml').write_text(yaml.safe_dump(descriptor, allow_unicode=True, sort_keys=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT / 'data' / 'visdrone_yolo'))
    args = parser.parse_args()
    convert(dataset_root(), Path(args.output))
    print('YOLO annotations exported; original annotations preserved.')

if __name__ == '__main__':
    main()
