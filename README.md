[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project - Классификация музыкального жанра по аудио-характеристикам и текстовым признакам

**Студент:** Барайщук Константин Аркадьевич

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Проверка кода](#проверка-кода-линтеры)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Классификация музыкального жанра (~28k треков, 1950–2019)

**Датасет:** [Music Dataset: 1950 to 2019](https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019)

**Признаки:** аудио-характеристики (danceability, energy, loudness и др.) + тематические вероятности из текста (16 топиков) + стилистические признаки lyrics + 9 взаимодействий — итого 36 признаков

**Целевая метрика:** Macro F1 (основная) + Accuracy — целевая переменная `genre` (7 классов: blues, country, hip hop, jazz, pop, reggae, rock)

**Обоснование метрики:** классы несбалансированы, Accuracy вводила бы в заблуждение; Macro F1 усредняет F1 по каждому жанру с равным весом


## Структура репозитория
```
.
├── data
│   ├── processed            # Очищенные и обработанные данные (train/val/test split)
│   └── raw                  # Исходные файлы (tcc_ceds_music.csv)
├── models                   # Сохранённые артефакты обученной модели
│   ├── best_model.pkl       # StackingClassifier v2 (финальная модель)
│   ├── feature_cols.pkl     # Список 36 признаков в нужном порядке
│   └── scaler.pkl           # StandardScaler (использовался для PCA/UMAP, не для инференса)
├── notebooks
│   ├── 01_eda.ipynb         # Разведочный анализ данных (EDA)
│   ├── 02_baseline.ipynb    # Baseline-модели (DummyClassifier, LogReg, KNN)
│   └── 03_experiments.ipynb # Эксперименты (11 моделей + feature engineering + ансамбли)
├── report
│   ├── images               # Изображения для отчёта
│   └── report.md            # Финальный отчёт (нет)
├── src
│   ├── __init__.py
│   ├── preprocessing.py     # Вспомогательные функции предобработки
│   └── modeling.py          # Обучение и оценка моделей
├── tests
│   └── test.py              # Тесты пайплайна
├── Dockerfile               # Образ для Jupyter (все зависимости)
├── docker-compose.yml       # Запуск Jupyter-сервера в контейнере
├── Makefile                 # lint / format / test
├── requirements.txt
└── README.md
```

## Запуск

### Вариант 1 — локально

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-konstantinba
cd hseml-group-project-konstantinba

# 2. Создать виртуальное окружение
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Загрузить данные NLTK для textblob
python -m textblob.download_corpora

# 5. Запустить Jupyter
jupyter notebook
# Запускать ноутбуки по порядку: 01 → 02 → 03
```

### Вариант 2 — Docker

```bash
# Собрать и запустить контейнер
docker compose up --build

# Jupyter доступен по адресу: http://localhost:8888
```

## Данные
- `data/raw/` — исходный файл `tcc_ceds_music.csv` (~28k строк, не включён в репозиторий)
- `data/processed/` — предобработанные файлы: `train.csv`, `val.csv`, `test.csv` (стратифицированный сплит 70/15/15)


## Результаты

### CP1 - Базовые модели (val / test)

| Модель | Macro F1 (val) | Accuracy (val) | Примечание |
|--------|---------------|----------------|------------|
| DummyClassifier | 0.1419 | 0.1713 | Нижняя граница |
| KNN (k=5) | 0.3660 | 0.3593 | Без настройки |
| LogisticRegression | 0.4161 | 0.4077 | Лучшая базовая |

LogisticRegression на тесте: Macro F1 = **0.4228**, Accuracy = 0.4088

### CP2 - Эксперименты (val / test)

| № | Модель | Macro F1 (val) | Macro F1 (test) | Примечание |
|---|--------|---------------|-----------------|------------|
| baseline | LogisticRegression | 0.4161 | 0.4228 | CP1, нижняя граница |
| 1 | RandomForest | 0.4724 | — | Без настройки |
| 2 | LightGBM | 0.4940 | — | Без настройки |
| 3 | LinearSVC | 0.4077 | — | Линейная граница недостаточна |
| 4 | ExtraTrees | 0.4699 | — | Схожее с RF |
| 5а | RF (GridSearchCV) | 0.4909 | — | class_weight=balanced, n_estimators=300 |
| 5б | LightGBM (GridSearchCV) | 0.4974 | — | num_leaves=63, lr=0.1 |
| 6 | PCA + LogReg | — | — | Снижает качество, от 15 компонент |
| 7 | VotingClassifier (soft) | 0.5038 | 0.5106 | RF + LightGBM + LogReg |
| 8 | StackingClassifier v1 (27 признаков) | 0.5106 | 0.5063 | RF+LGB+ET -> LogReg |
| 9 | LightGBM v2 (регуляризация) | 0.4993 | — | subsample=0.8, feature_fraction=0.8 |
| **10** | **StackingClassifier v2 (36 признаков)** | **0.5123** | **0.5076** | LGB v2 + таргет. признаки blues/rock |

**Финальная модель:** StackingClassifier v2 (RF + LightGBM v2 + ExtraTrees -> LogReg мета-классификатор)

- **Прирост над baseline:** +8.48% Macro F1 (test): 0.4228 -> 0.5076
- **Лучший жанр:** `hip hop` (F1 = 0.64) - хорошо выделяется по текстовым признакам
- **Сложные жанры:** `blues` и `rock` (F1 = 0.40) - высокое пересечение с country/pop по аудио-профилю
- **36 признаков:** 27 базовых + 9 взаимодействий (`sadness_acoustic`, `energy_nondance`, `valence_age` и др.)


## Проверка кода (линтеры)

```bash
# (не Windows)
make lint
# или вручную:
ruff check src/ tests/ notebooks/      # включая .ipynb
flake8 src/ tests/ --max-line-length=120
```