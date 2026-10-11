# Design-Code AI Measurement Service

Python FastAPI сървър за AI измерване на помещения от снимки.

## Инсталация

```bash
cd design-code-website/ai-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Забележка: `segment-anything` се инсталира директно от GitHub. Ако имате проблеми, махнете го временно от `requirements.txt` — модулът се използва само в по-късните фази.

## Стартиране

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Сървърът ще е достъпен на `http://localhost:8000`.

## Endpoints

- `GET /health` — проверка дали сървърът работи.
- `POST /analyze` — приема base64 изображение и връща измерени размери.

### Примерно тяло на заявка

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUg...",
  "marker_type": "a4",
  "custom_size_cm": 29.7
}
```

### Примерен отговор

```json
{
  "width_cm": 420.0,
  "height_cm": 260.0,
  "length_cm": 360.0,
  "px_per_cm": 12.5,
  "marker_label": "A4 лист",
  "confidence": 0.62,
  "demo_model_url": "https://threejs.org/examples/models/gltf/DamagedHelmet/glTF/DamagedHelmet.gltf",
  "note": "Това е демонстрационно измерване..."
}
```

## Следващи стъпки

1. Интеграция на SAM за сегментация на под/стени.
2. Интеграция на MiDaS или Depth Anything за оценка на дълбочина.
3. Детекция на мащабен маркер чрез OpenCV/YOLO.
4. Изчисляване на vanishing points за коректна перспектива.
