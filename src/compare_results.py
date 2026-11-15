"""Aggregate stage-7 metrics.json files into CSV. Run: python -m src.compare_results"""
import csv
import json
from pathlib import Path
from src.common import ROOT


def main():
    folder = ROOT / 'outputs' / 'comparisons'
    rows = []
    for path in sorted(folder.glob('*/metrics.json')):
        result = json.loads(path.read_text(encoding='utf-8'))
        row = {k: result.get(k) for k in ('model', 'split', 'images', 'imgsz', 'mean_ms_per_image', 'fps', 'peak_cuda_allocated_mb')}
        row.update(result.get('metrics', {}))
        row['directory'] = str(path.parent)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No evaluation results in {folder}; run src.evaluate first')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    output = folder / 'comparison.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output}')


if __name__ == '__main__':
    main()
