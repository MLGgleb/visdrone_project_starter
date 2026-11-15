"""Shared training configuration, reproducibility and metadata."""
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import yaml
from src.common import ROOT


def read_config(name):
    path = ROOT / 'configs' / 'models' / f'{name}.yaml'
    with path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_data():
    root = ROOT / 'data' / 'prepared'
    for path in (root/'dataset.yaml', root/'manifest.json', root/'annotations/instances_train.json', root/'annotations/instances_val.json'):
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}. Run python -m src.prepare_pipeline first.')
    return root


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metadata(name, cfg, output):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'data/prepared/manifest.json'
    details = {
        'model': name, 'config': cfg, 'dataset_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'started_at_unix': time.time(),
    }
    try:
        details['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        details['git_commit'] = None
    (output/'run_metadata.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
    return details
