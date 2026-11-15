"""Unified COCO evaluation for YOLO11s, RT-DETR-L and torchvision Faster R-CNN.
Run: python -m src.evaluate --model yolo11s --split val
"""
import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from src.common import ROOT

MODEL_NAMES = ('yolo11s', 'rtdetr_l', 'faster_rcnn')
METRIC_NAMES = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large')


def default_weights(model):
    base = ROOT / 'outputs' / 'experiments' / model
    if model == 'faster_rcnn':
        return base / 'last.pt'
    return base / 'weights' / 'best.pt'


def load_model(model_name, weights, device):
    if model_name in ('yolo11s', 'rtdetr_l'):
        from ultralytics import YOLO, RTDETR
        return (YOLO if model_name == 'yolo11s' else RTDETR)(str(weights))
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from src.training.common import read_config
    cfg = read_config('faster_rcnn')
    net = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                 min_size=cfg['short_side'], max_size=cfg['max_size'])
    n = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(n, 11)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['model'])
    return net.to(device).eval()


def predict(model_name, model, image_path, device, imgsz, conf, max_det):
    if model_name != 'faster_rcnn':
        results = model.predict(source=str(image_path), device=device, imgsz=imgsz,
                                conf=conf, iou=0.7, max_det=max_det, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        cats = boxes.cls.cpu().tolist()
        return [(b, float(s), int(c)+1) for b, s, c in zip(xyxy, scores, cats)]
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(image_path) as im:
        image = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2, 0, 1).float().div_(255).to(device)
    with torch.inference_mode():
        out = model([image])[0]
    boxes = out['boxes'].cpu().tolist()
    scores = out['scores'].cpu().tolist()
    cats = out['labels'].cpu().tolist()
    return [(b, float(s), int(c)) for b, s, c in zip(boxes, scores, cats) if s >= conf][:max_det]


def evaluate(model_name, split='val', weights=None, imgsz=640, conf=0.001, max_det=300,
             device='cuda:0', limit=None, output=None):
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu or install CUDA-enabled PyTorch')
    root = ROOT / 'data' / 'prepared'
    annotation_file = root / 'annotations' / f'instances_{split}.json'
    if not annotation_file.exists():
        raise FileNotFoundError(annotation_file)
    weights = Path(weights) if weights else default_weights(model_name)
    if not weights.exists():
        raise FileNotFoundError(f'Weights not found: {weights}. Train the model first or use --weights.')
    gt = COCO(str(annotation_file))
    images = sorted(gt.dataset['images'], key=lambda x: x['id'])
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be >= 1')
        images = images[:limit]
    model = load_model(model_name, weights, device)
    predictions = []
    timings = []
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for index, info in enumerate(images, 1):
        path = root / 'images' / split / info['file_name']
        if not path.exists():
            raise FileNotFoundError(path)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        detections = predict(model_name, model, path, device, imgsz, conf, max_det)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        for box, score, cat in detections:
            x1, y1, x2, y2 = box
            w, h = max(0.0, x2-x1), max(0.0, y2-y1)
            if w > 0 and h > 0 and 1 <= cat <= 10:
                predictions.append({'image_id': info['id'], 'category_id': cat,
                                    'bbox': [x1, y1, w, h], 'score': score})
        if index % 50 == 0:
            print(f'Processed {index}/{len(images)} images', flush=True)
    output = Path(output) if output else ROOT / 'outputs' / 'comparisons' / f'{model_name}_{split}_{imgsz}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'predictions.json').write_text(json.dumps(predictions), encoding='utf-8')
    # COCO.loadRes cannot load an empty detection list in some pycocotools versions.
    metrics = {key: None for key in METRIC_NAMES}
    if predictions:
        detections = gt.loadRes(predictions)
        evaluator = COCOeval(gt, detections, 'bbox')
        evaluator.params.imgIds = [info['id'] for info in images]
        evaluator.params.maxDets = [1, 10, max_det]
        evaluator.evaluate()
        evaluator.accumulate()
        # pycocotools.summarize() hardcodes maxDets=100 for its first AP row.
        # With VisDrone's maxDets=300, that row would incorrectly be -1.
        # Compute AP@[.50:.95] explicitly at the requested maxDets.
        import numpy as np
        precision = evaluator.eval['precision']  # T x R x K x A x M
        max_det_index = evaluator.params.maxDets.index(max_det)
        all_area_index = evaluator.params.areaRngLbl.index('all')
        ap_values = precision[:, :, :, all_area_index, max_det_index]
        valid_ap = ap_values[ap_values > -1]
        ap_50_95 = float(np.mean(valid_ap)) if valid_ap.size else None

        # Keep the standard COCOeval AP50/AP75/size-specific AP and AR rows.
        # Hide only the misleading hardcoded first line of summarize().
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            evaluator.summarize()
        summary_lines = captured.getvalue().splitlines()
        if summary_lines:
            print(' Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all '
                  f'| maxDets={max_det:3d} ] = '
                  f'{ap_50_95:.3f}' if ap_50_95 is not None else
                  ' Average Precision (AP) = undefined')
            print('\n'.join(summary_lines[1:]))
        metrics = {name: (float(v) if v >= 0 else None)
                   for name, v in zip(METRIC_NAMES, evaluator.stats)}
        metrics['AP'] = ap_50_95
    else:
        print('No predictions; COCO AP is undefined for this run.')
    # Timing includes image loading and framework postprocessing, not just GPU kernels.
    measured = timings[5:] if len(timings) > 5 else timings
    mean_s = sum(measured) / len(measured) if measured else None
    result = {'model': model_name, 'split': split, 'images': len(images),
              'weights': str(weights.resolve()), 'imgsz': imgsz,
              'conf': conf, 'max_det': max_det, 'device': device,
              'metrics': metrics, 'predictions': len(predictions),
              'mean_ms_per_image': round(mean_s * 1000, 3) if mean_s else None,
              'fps': round(1 / mean_s, 3) if mean_s else None,
              'timing_scope': 'single-image end-to-end predict call; first five warm-up images excluded',
              'peak_cuda_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2)
              if device.startswith('cuda') else None}
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODEL_NAMES, required=True)
    p.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    p.add_argument('--weights', type=Path)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, help='Smoke test only; not comparable with full val')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    evaluate(a.model, a.split, a.weights, a.imgsz, a.conf, a.max_det, a.device, a.limit, a.output)


if __name__ == '__main__':
    main()
