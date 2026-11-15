"""Download and safely extract the official VisDrone DET archives.
Run: python -m src.download_data --splits train val test
"""
import argparse
import shutil
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
from .common import dataset_root, SPLITS

BASE = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/'

def download(url: str, dest: Path):
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f'Using existing archive: {dest}')
        return
    tmp = dest.with_suffix(dest.suffix + '.part')
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(20, 120)) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with tmp.open('wb') as out, tqdm(total=total or None, unit='B', unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        bar.update(len(chunk))
        if not zipfile.is_zipfile(tmp):
            raise ValueError(f'Invalid ZIP archive: {tmp}')
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

def extract_safe(archive: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = (base / member.filename).resolve()
            if not path.is_relative_to(base):
                raise ValueError(f'Unsafe archive entry: {member.filename}')
            # Avoid symlink entries in downloaded archives.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink in archive: {member.filename}')
        z.extractall(base)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    parser.add_argument('--keep-archives', action='store_true')
    args = parser.parse_args()
    root = dataset_root()
    for split in args.splits:
        name = SPLITS[split]
        archive = root / 'archives' / f'{name}.zip'
        target = root / name
        if (target / 'images').is_dir() and (target / 'annotations').is_dir():
            print(f'Already extracted: {name}')
            continue
        download(BASE + archive.name, archive)
        extract_safe(archive, root)
        if not (target / 'images').is_dir() or not (target / 'annotations').is_dir():
            raise RuntimeError(f'Expected structure not found after extraction: {target}')
        print(f'Extracted {name}')
        if not args.keep_archives:
            archive.unlink()

if __name__ == '__main__':
    main()
