"""Train YOLO11s or RT-DETR-L using the same prepared YOLO dataset."""
import argparse
import json
import time
from pathlib import Path
from src.common import ROOT
from .common import check_data, metadata, read_config, set_seed


def train(name, dry_run=False):
    if name not in ('yolo11s', 'rtdetr_l'):
        raise ValueError(name)
    root = check_data()
    cfg = read_config(name)
    output = ROOT / 'outputs/experiments' / name
    if dry_run:
        return {'model': name, 'data': str(root/'dataset.yaml'), 'config': cfg, 'output': str(output)}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install CUDA-enabled PyTorch; check torch.cuda.is_available().')
    from ultralytics import YOLO, RTDETR
    set_seed(cfg['seed'])
    output.mkdir(parents=True, exist_ok=True)
    metadata(name, cfg, output)
    model = YOLO(cfg['model']) if name == 'yolo11s' else RTDETR(cfg['model'])
    opts = {k: v for k, v in cfg.items() if k != 'model'}
    started = time.perf_counter()
    model.train(data=str(root/'dataset.yaml'), device=0, project=str(output.parent), name=name,
                exist_ok=True, **opts)
    elapsed = time.perf_counter() - started
    summary = {'model': name, 'seconds': elapsed, 'best_weights': str(output/'weights/best.pt')}
    (output/'training_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['yolo11s', 'rtdetr_l'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(train(args.model, args.dry_run), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
