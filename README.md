[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Классификация музыкального жанра по аудио-характеристикам и текстовым признакам

**Студент:** Барайщук Константин Аркадьевич

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

<!-- Кратко опишите задачу: что предсказываем, какой датасет, метрика качества -->

**Задача:** Классификация музыкального жанра (~28k треков, 1950–2019)

**Датасет:** [Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)

**Признаки:** аудио-характеристики (danceability, energy, loudness и др.) + тематические вероятности из текста + стилистические признаки фрагмента lyrics

**Целевая метрика:** Macro F1 (основная) + Accuracy - целевая переменная `genre`

**Обоснование метрики:** классы несбалансированы, Accuracy вводила бы в заблуждение; Macro F1 усредняет F1 по каждому жанру с равным весом


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные (train/val/test)
│   └── raw                     # Исходные файлы (tcc_ceds_music.csv)
├── models                      # Сохранённые модели (пока нет)
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модели (DummyClassifier, LogReg, KNN)
│   └── 03_experiments.ipynb    # Эксперименты (позже)
├── presentation                # Презентация для защиты (пока нет)
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт (пока нет)
├── src
│   ├── __init__.py
│   ├── preprocessing.py        # Доп функции
│   └── modeling.py             # Обучение и оценка моделей (пока нет)
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

### Вариант 1 — локально

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-konstantinba/tree/cp1
cd hseml-group-project-konstantinba

# 2. Создать виртуальное окружение
python -m venv venv
source .venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Загрузить данные NLTK для textblob
python -m textblob.download_corpora

# 5. Запустить Jupyter
jupyter notebook
# run all в ноутбуках (1 потом 2)
```

### Вариант 2 — Docker (пока нет)

```bash
# Собрать и запустить контейнер
docker compose up --build

# Jupyter будет доступен по адресу: http://localhost:8888
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты

### CP1 - Базовые модели (val / test)

| Модель | Macro F1 (val) | Accuracy (val) | Примечание |
|--------|---------------|----------------|------------|
| DummyClassifier | 0.1419 | 0.1713 | Нижняя граница |
| KNN (k=5) | 0.3660 | 0.3593 | Без настройки |
| LogisticRegression | 0.4161 | 0.4077 | Лучшая на данный мосент |

LogisticRegression на тесте: Macro F1 = 0.4228, Accuracy = 0.4088

Лучший по классам: `hip hop` (F1=0.55). Наиболее сложный: `blues` (F1=0.27).


## Отчёт (пока нет)

Финальный отчёт: [`report/report.md`](report/report.md)
