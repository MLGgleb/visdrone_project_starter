# Этап 10. Контейнеризация FastAPI + YOLO11s

## Что контейнеризовано

Сервис из `service/app.py` (веб-страница, REST API, изображения и видео). Обучение моделей и датасет **не** входят в образ. Веса подаются через bind mount, история и обработанные файлы сохраняются в именованном Docker volume.

## Подготовка

1. Установить Docker Desktop (Windows: режим Linux containers, WSL2). Для GPU также нужны драйвер NVIDIA, поддержка GPU в WSL2/Docker Desktop и работающий NVIDIA Container Toolkit / GPU runtime.
2. В корне проекта поместить обученные веса в `models/final/yolo11s_best.pt` (веса не входят в ZIP).
3. Проверить, что Docker запущен: `docker version` и `docker compose version`.

### CPU (первый запуск / проверка)

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f visdrone
```

Открыть http://127.0.0.1:8000 и http://127.0.0.1:8000/docs .

Проверка API:

```bash
curl http://127.0.0.1:8000/api/health
curl -F "file=@sample.jpg" -F "confidence=0.25" http://127.0.0.1:8000/api/detect/image
```

В PowerShell используйте `curl.exe` вместо `curl`, например `curl.exe http://127.0.0.1:8000/api/health`.

### GPU (RTX 4060 Ti)

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
docker compose -f compose.yaml -f compose.gpu.yaml logs -f visdrone
```

В GPU-сборке PyTorch устанавливается из официального CUDA 12.6 wheel-индекса. GPU-драйвер и совместимость CUDA должны быть обеспечены на хосте. Если GPU не виден контейнеру, сначала проверьте `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi` (при наличии доступа к образу). Не используйте одновременно CPU и GPU Compose-конфигурации для одного сервиса.

### Хранение данных

- Модель: `./models/final/yolo11s_best.pt` → `/app/models/final/yolo11s_best.pt` (только чтение).
- Результаты и SQLite: volume `visdrone_results` → `/app/outputs/service`.
- `docker compose down` останавливает контейнер и **сохраняет** volume.
- `docker compose down -v` **удаляет историю и результаты**.

### Диагностика

- `GET /api/health` показывает `weights_exist` и выбранное устройство, но не проверяет фактический успешный инференс.
- Если `weights_exist=false`, убедитесь, что файл находится в `models/final/` на хосте.
- Если контейнер работает, но первый запрос медленный, это может быть связано с первой загрузкой весов.
- Если ошибка `CUDA is not available`, проверьте GPU runtime, драйвер и GPU Compose-конфигурацию.
- Для небольшого GPU (8 ГБ) используйте один процесс uvicorn (`--workers 1`).
- Результат MP4 зависит от наличия кодека `mp4v` в установленном OpenCV/FFmpeg; отдельно проверьте видео на вашей машине.

### Остановка

```bash
docker compose down
```

## Проверки

Синтаксис Python и YAML проверен локально. Полная сборка Docker и реальный GPU-инференс в данной среде не запускались; проверьте их на своём ПК.
