from pathlib import Path
from PIL import Image
from src.analyze_data import inspect, parse_annotation
from src.common import SPLITS, config


def test_annotation_parser(tmp_path):
    p = tmp_path / 'sample.txt'
    p.write_text('1,2,10,20,1,4,0,2\ninvalid\n', encoding='utf-8')
    rows = list(parse_annotation(p))
    assert rows[0][1] == [1, 2, 10, 20, 1, 4, 0, 2]
    assert rows[1][2] is not None


def test_audit_finds_invalid_box(tmp_path):
    for folder in SPLITS.values():
        (tmp_path / folder / 'images').mkdir(parents=True)
        (tmp_path / folder / 'annotations').mkdir()
    folder = tmp_path / SPLITS['train']
    Image.new('RGB', (32, 32)).save(folder / 'images' / 'example.jpg')
    (folder / 'annotations' / 'example.txt').write_text('30,30,10,10,1,4,0,0\n', encoding='utf-8')
    cfg = config()
    cfg['expected_images'] = {'train': 1, 'val': 0, 'test': 0}
    images, objects, errors = inspect(tmp_path, cfg)
    assert len(images) == 1
    assert objects.empty
    assert 'invalid/out-of-image bounding box' in errors['error'].tolist()
