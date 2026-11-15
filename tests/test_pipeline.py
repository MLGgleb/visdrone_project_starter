import json
import numpy as np
import pytest
from PIL import Image
from src.prepare_pipeline import annotations, prepare
from src.transforms import transform
from src.common import SPLITS


def make_fixture(tmp_path):
    root = tmp_path / 'raw'
    for i, folder in enumerate(SPLITS.values()):
        (root / folder / 'images').mkdir(parents=True)
        (root / folder / 'annotations').mkdir()
        Image.new('RGB', (100, 50), (i*50, 0, 0)).save(root / folder / 'images' / f'{i}.jpg')
        (root / folder / 'annotations' / f'{i}.txt').write_text('10,5,20,10,1,4,0,0\n0,0,10,10,0,0,0,0\n')
    return root


def test_prepare(tmp_path):
    raw = make_fixture(tmp_path)
    output = tmp_path / 'prepared'
    result = prepare(raw, output, method='copy')
    assert result['splits']['train']['objects'] == 1
    assert (output / 'dataset.yaml').exists()
    assert (output / 'labels' / 'train' / '0.txt').read_text().startswith('3 ')
    coco = json.loads((output / 'annotations' / 'instances_train.json').read_text())
    assert coco['annotations'][0]['bbox'] == [10, 5, 20, 10]


def test_invalid_bbox(tmp_path):
    path = tmp_path / 'bad.txt'
    path.write_text('99,0,20,10,1,4,0,0\n')
    with pytest.raises(ValueError):
        annotations(path, 100, 50)


def test_deterministic_eval():
    im = Image.new('RGB', (100, 50), 'white')
    boxes = [[10, 5, 30, 15]]
    a, ba = transform(im, boxes, train=False, seed=1)
    b, bb = transform(im, boxes, train=False, seed=9)
    assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(ba, bb)
    np.testing.assert_allclose(ba[0], [64, 192, 192, 256])


def test_cross_split_duplicate(tmp_path):
    raw = make_fixture(tmp_path)
    train_image = next((raw / SPLITS['train'] / 'images').glob('*.jpg'))
    val_image = next((raw / SPLITS['val'] / 'images').glob('*.jpg'))
    val_image.write_bytes(train_image.read_bytes())
    with pytest.raises(ValueError, match='Cross-split'):
        prepare(raw, tmp_path / 'prepared', method='copy')
