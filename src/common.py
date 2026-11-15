from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def config():
    with (ROOT / 'configs/data.yaml').open(encoding='utf-8') as f:
        return yaml.safe_load(f)

def dataset_root():
    return ROOT / config()['dataset_root']

SPLITS = {'train': 'VisDrone2019-DET-train', 'val': 'VisDrone2019-DET-val', 'test': 'VisDrone2019-DET-test-dev'}
