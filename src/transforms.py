"""Deterministic validation/test and bbox-aware training augmentation for custom PyTorch datasets.
This module is optional: YOLO/RT-DETR training libraries use their own transforms.
"""
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def transform(image, boxes, train=False, seed=None, size=640):
    """Return RGB PIL image and XYXY pixel boxes; preserve aspect ratio by padding.
    boxes: array-like [N, 4], original pixel coordinates.
    """
    rng = random.Random(seed)
    image = image.convert('RGB')
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    width, height = image.size
    if train:
        if rng.random() < 0.5:
            image = ImageOps.mirror(image)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if rng.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        if rng.random() < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))
    scale = min(size / width, size / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    left, top = (size-new_width)//2, (size-new_height)//2
    result = Image.new('RGB', (size, size), (114, 114, 114))
    result.paste(resized, (left, top))
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, size)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, size)
    return result, boxes
