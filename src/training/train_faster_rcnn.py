"""Train torchvision Faster R-CNN ResNet50-FPN on COCO-formatted VisDrone."""
import argparse
import json
import time
from pathlib import Path
from PIL import Image
from src.common import ROOT
from .common import check_data, metadata, read_config, set_seed


class VisDroneCoco:
    def __init__(self, root, split):
        import torch
        self.root = Path(root)
        self.split = split
        coco = json.loads((self.root/'annotations'/f'instances_{split}.json').read_text(encoding='utf-8'))
        self.images = coco['images']
        self.boxes = {}
        for a in coco['annotations']:
            self.boxes.setdefault(a['image_id'], []).append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        import torch
        import numpy as np
        info = self.images[index]
        image = Image.open(self.root/'images'/self.split/info['file_name']).convert('RGB')
        arr = np.array(image, copy=True)
        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255)
        anns = self.boxes.get(info['id'], [])
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]] for a in anns]
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.tensor([a['category_id'] for a in anns], dtype=torch.int64),
            'image_id': torch.tensor(info['id'], dtype=torch.int64),
            'area': torch.tensor([a['area'] for a in anns], dtype=torch.float32),
            'iscrowd': torch.zeros(len(anns), dtype=torch.int64),
        }
        return img, target


def collate(batch):
    return tuple(zip(*batch))


def train(dry_run=False):
    root = check_data()
    cfg = read_config('faster_rcnn')
    output = ROOT/'outputs/experiments/faster_rcnn'
    if dry_run:
        return {'model': 'faster_rcnn', 'config': cfg, 'data': str(root), 'output': str(output)}
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch.')
    set_seed(cfg['seed'])
    output.mkdir(parents=True, exist_ok=True)
    metadata('faster_rcnn', cfg, output)
    train_ds = VisDroneCoco(root, 'train')
    val_ds = VisDroneCoco(root, 'val')
    train_loader = DataLoader(train_ds, batch_size=cfg['batch'], shuffle=True, num_workers=cfg['workers'], collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg['workers'], collate_fn=collate)
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                    min_size=cfg['short_side'], max_size=cfg['max_size'])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)  # background + 10 classes
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['step_size'], gamma=cfg['gamma'])
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    started = time.perf_counter()
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        running, count = 0., 0
        for images, targets in train_loader:
            images = [im.cuda(non_blocking=True) for im in images]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
            if cfg['horizontal_flip_probability']:
                import random
                for i, im in enumerate(images):
                    if random.random() < cfg['horizontal_flip_probability']:
                        width = im.shape[-1]
                        images[i] = im.flip(-1)
                        b = targets[i]['boxes']
                        if len(b):
                            targets[i]['boxes'] = torch.stack([width-b[:, 2], b[:, 1], width-b[:, 0], b[:, 3]], dim=1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=cfg['amp']):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().item())
            count += 1
        scheduler.step()
        entry = {'epoch': epoch+1, 'train_loss': running/max(count, 1), 'lr': optimizer.param_groups[0]['lr']}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch+1, 'config': cfg}, output/'last.pt')
        (output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    elapsed = time.perf_counter()-started
    result = {'model': 'faster_rcnn', 'seconds': elapsed, 'last_weights': str(output/'last.pt'), 'validation_images': len(val_ds), 'validation_metrics': 'not computed; use common evaluation in stage 7'}
    (output/'training_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(train(args.dry_run), indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
