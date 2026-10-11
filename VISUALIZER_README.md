# Design-Code 3D Визуализатор

Браузърен инструмент за AI измерване на помещения и 3D визуализация на интериорни дизайни.

## Какво включва (MVP)

- Качване на снимка на помещение.
- AI измерване на размери (демо евристика + опция за реални AI модели).
- Мащабен маркер: A4 лист, кутия цигари, банкнота или ръчно въвеждане.
- 3D визуализация на стаята с Three.js.
- Каталог с демо 3D модели.
- Ръчно коригиране на размери.
- Експорт на изображение от 3D сцената.
- Технически панел с изчислени размери.

## Как се пуска локално

### 1. Frontend (сайт)

```bash
cd design-code-website
node server.js
```

Сайтът ще е на `http://localhost:8765/`. Отвори `http://localhost:8765/visualizer.html`.

### 2. Backend (AI сървър)

```bash
cd design-code-website/ai-service
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

AI сървърът ще е на `http://localhost:8000/`.

> Забележка: `requirements.txt` използва `opencv-python-headless`. Ако имаш проблеми с NumPy, инсталирай `numpy<2`:
> `pip install "numpy<2" --force-reinstall`

### 3. Тествай flow-а

1. Отвори `http://localhost:8765/visualizer.html`.
2. Качи снимка на стая.
3. Избери мащабен маркер.
4. Натисни "Анализирай снимката".
5. Виж размерите и 3D сцената.
6. Добави демо модел от каталога.
7. Коригирай размерите ръчно, ако е необходимо.
8. Експортирай изображение.

## Структура на файловете

```
design-code-website/
├── visualizer.html          # Главна страница на визуализатора
├── project.css              # Споделени стилове
├── index.html               # Добавен линк към визуализатора
├── ASSETS_GUIDE.md          # Инструкции за дизайнерката за 3D модели
├── VISUALIZER_README.md     # Този файл
└── ai-service/
    ├── main.py              # FastAPI сървър
    ├── ai_models.py         # Опционални AI модели (SAM, Depth)
    ├── requirements.txt     # Python зависимости
    └── README.md            # Инструкции за AI сървъра
```

## Настройка за тестова/продукционна среда

В `visualizer.html` промени URL-а на AI сървъра:

```javascript
const res = await fetch('http://localhost:8000/analyze', { ... });
```

За тестова/продукционна среда замени `http://localhost:8000/analyze` с реалния URL, например:

```javascript
const res = await fetch('https://ai-test.design-code.bg/analyze', { ... });
```

Препоръчително е URL-ът да се изнася в environment variable или config файл, а не да е hardcoded.

## AI модели (експериментално)

За да се активират реалните AI модели (SAM + Depth Anything/MiDaS):

1. Инсталирай допълнителните зависимости:

```bash
pip install torch torchvision transformers
pip install git+https://github.com/facebookresearch/segment-anything.git
```

2. Изтегли SAM checkpoint от https://github.com/facebookresearch/segment-anything#model-checkpoints и го сложи в `ai-service/`.

3. В `visualizer.html` маркирай "Използвай AI модели" преди анализ.

> Внимание: първото зареждане на моделите изтегля големи файлове и отнема време.

## Следващи стъпки

1. Дизайнерката подготвя реални GLB модели според `ASSETS_GUIDE.md`.
2. Замяна на demo моделите с реални мебели.
3. Добавяне на photo overlay режим.
4. Генериране на PDF технически план.
5. Технически панел за корекция от техници.
6. Deploy на тестов сървър.
