"""FastAPI demo for the selected YOLO11s VisDrone detector."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from src.common import ROOT

WEIGHTS = Path(os.getenv('VISDRONE_WEIGHTS', str(ROOT / 'models' / 'final' / 'yolo11s_best.pt')))
STORAGE = Path(os.getenv('VISDRONE_STORAGE', str(ROOT / 'outputs' / 'service')))
MAX_UPLOAD_MB = int(os.getenv('VISDRONE_MAX_UPLOAD_MB', '100'))
MAX_VIDEO_FRAMES = int(os.getenv('VISDRONE_MAX_VIDEO_FRAMES', '900'))
DEVICE = os.getenv('VISDRONE_DEVICE', 'cuda:0')
CLASS_NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
               'tricycle', 'awning-tricycle', 'bus', 'motor')
LOCK = threading.Lock()
MODEL = None


def db_connect():
    con = sqlite3.connect(STORAGE / 'history.sqlite3', timeout=30)
    con.row_factory = sqlite3.Row
    return con


def initialize_storage():
    STORAGE.mkdir(parents=True, exist_ok=True)
    with db_connect() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS history (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL,
            original_name TEXT NOT NULL, output_name TEXT NOT NULL,
            detections INTEGER NOT NULL, elapsed_seconds REAL NOT NULL,
            confidence REAL NOT NULL, frames INTEGER NOT NULL)''')


def get_model():
    global MODEL
    if MODEL is None:
        if not WEIGHTS.is_file():
            raise RuntimeError(f'Model weights not found: {WEIGHTS}. Copy your trained best.pt to models/final/yolo11s_best.pt')
        from ultralytics import YOLO
        MODEL = YOLO(str(WEIGHTS))
    return MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_storage()
    yield


app = FastAPI(title='VisDrone YOLO11s Demo', version='1.0', lifespan=lifespan)
app.mount('/results', StaticFiles(directory=str(STORAGE), check_dir=False), name='results')


def valid_confidence(value: float):
    if not 0.01 <= value <= 1.0:
        raise HTTPException(422, 'confidence must be between 0.01 and 1.0')


async def read_upload(upload: UploadFile, allowed: set[str]):
    ext = Path(upload.filename or '').suffix.lower()
    if ext not in allowed:
        raise HTTPException(415, f'Unsupported file type: {ext}')
    data = bytearray()
    while chunk := await upload.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f'File exceeds {MAX_UPLOAD_MB} MB')
    if not data:
        raise HTTPException(400, 'Empty file')
    return bytes(data), ext


def detect(frame: np.ndarray, confidence: float):
    with LOCK:
        result = get_model().predict(frame, conf=confidence, imgsz=640,
                                     device=DEVICE, max_det=300, verbose=False)[0]
    annotated = result.plot()
    items = []
    if result.boxes is not None:
        for xyxy, score, category in zip(result.boxes.xyxy.cpu().tolist(),
                                         result.boxes.conf.cpu().tolist(),
                                         result.boxes.cls.cpu().tolist()):
            index = int(category)
            items.append({'class_id': index, 'class_name': CLASS_NAMES[index] if index < 10 else str(index),
                          'confidence': round(float(score), 4),
                          'xyxy': [round(float(v), 2) for v in xyxy]})
    return annotated, items


def record(job_id, kind, filename, output_name, detections, elapsed, confidence, frames):
    with db_connect() as con:
        con.execute('''INSERT INTO history VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)''',
                    (job_id, kind, filename, output_name, detections, elapsed, confidence, frames))


@app.get('/', response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / 'static' / 'index.html').read_text(encoding='utf-8')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'weights_exist': WEIGHTS.is_file(), 'model': 'YOLO11s', 'device': DEVICE}


@app.get('/api/history')
def history(limit: int = 30):
    if not 1 <= limit <= 100:
        raise HTTPException(422, 'limit must be 1..100')
    with db_connect() as con:
        rows = con.execute('SELECT * FROM history ORDER BY created_at DESC, rowid DESC LIMIT ?', (limit,)).fetchall()
    return [{'id': r['id'], 'created_at': r['created_at'], 'kind': r['kind'],
             'filename': r['original_name'], 'detections': r['detections'],
             'elapsed_seconds': r['elapsed_seconds'], 'confidence': r['confidence'],
             'frames': r['frames'], 'result_url': f"/results/{r['output_name']}"} for r in rows]


@app.post('/api/detect/image')
async def image_detection(file: UploadFile = File(...), confidence: float = Form(0.25)):
    valid_confidence(confidence)
    data, ext = await read_upload(file, {'.jpg', '.jpeg', '.png', '.webp'})
    import io
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img).convert('RGB')
            if img.width * img.height > 40_000_000:
                raise HTTPException(413, 'Image exceeds 40 megapixels')
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, 'Invalid image') from exc
    started = time.perf_counter()
    try:
        annotated, items = detect(frame, confidence)
    except Exception as exc:
        raise HTTPException(503, f'Inference failed: {exc}') from exc
    elapsed = time.perf_counter() - started
    job_id = uuid.uuid4().hex
    output_name = f'{job_id}.jpg'
    if not cv2.imwrite(str(STORAGE / output_name), annotated):
        raise HTTPException(500, 'Could not save result')
    record(job_id, 'image', Path(file.filename or 'image').name, output_name, len(items), elapsed, confidence, 1)
    return {'id': job_id, 'kind': 'image', 'detections': items, 'count': len(items),
            'elapsed_seconds': round(elapsed, 3), 'result_url': f'/results/{output_name}'}


@app.post('/api/detect/video')
async def video_detection(file: UploadFile = File(...), confidence: float = Form(0.25)):
    valid_confidence(confidence)
    data, ext = await read_upload(file, {'.mp4', '.avi', '.mov', '.mkv'})
    job_id = uuid.uuid4().hex
    input_path = STORAGE / f'{job_id}_input{ext}'
    output_name = f'{job_id}.mp4'
    output_path = STORAGE / output_name
    input_path.write_bytes(data)
    cap = None
    writer = None
    failed = False
    frames = 0
    total_detections = 0
    started = time.perf_counter()
    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(400, 'Could not decode video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        while frames < MAX_VIDEO_FRAMES:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                if w <= 0 or h <= 0 or w * h > 8_300_000:
                    raise HTTPException(413, 'Video resolution too large')
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise HTTPException(500, 'MP4 encoder unavailable in OpenCV')
            annotated, items = detect(frame, confidence)
            writer.write(annotated)
            frames += 1
            total_detections += len(items)
        if frames == 0:
            raise HTTPException(400, 'Video has no readable frames')
        if frames == MAX_VIDEO_FRAMES:
            ok, _ = cap.read()
            if ok:
                raise HTTPException(413, f'Video exceeds limit of {MAX_VIDEO_FRAMES} frames')
    except HTTPException:
        failed = True
        raise
    except Exception as exc:
        failed = True
        raise HTTPException(503, f'Video processing failed: {exc}') from exc
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        input_path.unlink(missing_ok=True)
        if failed:
            output_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    record(job_id, 'video', Path(file.filename or 'video').name, output_name,
           total_detections, elapsed, confidence, frames)
    return {'id': job_id, 'kind': 'video', 'frames': frames,
            'total_detections_across_frames': total_detections,
            'elapsed_seconds': round(elapsed, 3), 'result_url': f'/results/{output_name}'}
