"""
Модуль предобработки данных для задачи классификации музыкального жанра.
Функции используются в ноутбуках 01_eda.ipynb и 02_baseline.ipynb.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# ---------------------------------------------------------------------------
# Константы — списки признаков
# ---------------------------------------------------------------------------

AUDIO_FEATURES: list[str] = [
    "danceability",
    "loudness",
    "acousticness",
    "instrumentalness",
    "valence",
    "energy",
]

# Исходные названия тематических столбцов в CSV
_TOPIC_FEATURES_RAW: list[str] = [
    "dating",
    "violence",
    "world/life",
    "night/time",
    "shake the audience",
    "family/gospel",
    "romantic",
    "communication",
    "obscene",
    "music",
    "movement/places",
    "light/visual perceptions",
    "family/spiritual",
    "like/girls",
    "sadness",
    "feelings",
]

# Переименованные тематические столбцы (без спецсимволов)
TOPIC_FEATURES: list[str] = [
    re.sub(r"[ /]+", "_", col) for col in _TOPIC_FEATURES_RAW
]

# Текстовые признаки, извлекаемые из lyrics.
# Примечание: stretched_words_ratio оказался нулевым на всём датасете —
# предобработка текста удалила растянутые слова, поэтому признак исключён.
TEXT_FEATURES: list[str] = [
    "repetition_ratio",
    "avg_word_length",
    "sentiment_polarity",
]

# Дополнительные числовые признаки из метаданных
META_FEATURES: list[str] = ["len", "age"]


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет из CSV-файла.

    Parameters
    ----------
    path:
        Путь к CSV-файлу.

    Returns
    -------
    pd.DataFrame
        Загруженный датасет.
    """
    df = pd.read_csv(path, index_col=0)
    print(f"Загружено строк: {len(df)}, столбцов: {len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Очистка данных
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает датасет: переименовывает столбцы, удаляет дубликаты,
    обрабатывает пропуски.

    Шаги:
    1. Переименование тематических столбцов (/ и пробелы → _).
    2. Удаление дубликатов по паре (artist_name, track_name).
    3. Заполнение пропусков в lyrics пустой строкой.
    4. Приведение числовых столбцов к float.

    Parameters
    ----------
    df:
        Исходный датасет.

    Returns
    -------
    pd.DataFrame
        Очищенный датасет.
    """
    df = df.copy()

    # 1. Переименовываем тематические столбцы
    rename_map = {raw: clean for raw, clean in zip(_TOPIC_FEATURES_RAW, TOPIC_FEATURES)}
    df = df.rename(columns=rename_map)
    print(f"Переименованы столбцы: {rename_map}")

    # 2. Удаляем дубликаты по (artist_name, track_name)
    n_before = len(df)
    df = df.drop_duplicates(subset=["artist_name", "track_name"])
    n_removed = n_before - len(df)
    print(f"Удалено дубликатов: {n_removed} (осталось строк: {len(df)})")

    # 3. Заполняем пропуски в lyrics пустой строкой
    if "lyrics" in df.columns:
        n_missing_lyrics = df["lyrics"].isna().sum()
        df["lyrics"] = df["lyrics"].fillna("")
        if n_missing_lyrics > 0:
            print(f"Заполнено пропусков в lyrics: {n_missing_lyrics}")

    # 4. Приводим числовые столбцы к float
    numeric_cols = AUDIO_FEATURES + TOPIC_FEATURES + ["len", "age"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Feature Engineering — текстовые признаки
# ---------------------------------------------------------------------------


def _compute_text_features_for_row(text: str) -> dict[str, float]:
    """Вычисляет текстовые признаки для одной строки lyrics."""
    if not isinstance(text, str) or text.strip() == "":
        return {feat: np.nan for feat in TEXT_FEATURES}

    words = text.lower().split()
    if not words:
        return {feat: np.nan for feat in TEXT_FEATURES}

    # Доля повторяющихся слов (1 - лексическое разнообразие)
    repetition_ratio = 1.0 - len(set(words)) / len(words)

    # Средняя длина слова
    avg_word_length = sum(len(w) for w in words) / len(words)

    # Тональность (TextBlob работает даже на коротком тексте)
    try:
        sentiment_polarity = TextBlob(text).sentiment.polarity
    except Exception:
        sentiment_polarity = np.nan

    return {
        "repetition_ratio": repetition_ratio,
        "avg_word_length": avg_word_length,
        "sentiment_polarity": sentiment_polarity,
    }


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Извлекает признаки стиля текста из столбца lyrics.

    Признаки не зависят от полноты текста (работают на фрагменте):
    - repetition_ratio   — доля повторяющихся слов (маркер рефрена)
    - avg_word_length    — средняя длина слова
    - sentiment_polarity — тональность [-1, 1] по TextBlob

    Примечание: stretched_words_ratio был исключён после EDA —
    предобработка датасета удалила все растянутые слова, признак = 0.

    Parameters
    ----------
    df:
        Датасет с колонкой lyrics.

    Returns
    -------
    pd.DataFrame
        Датасет с добавленными текстовыми признаками.
    """
    df = df.copy()
    print("Вычисляем текстовые признаки из lyrics...")
    features = df["lyrics"].apply(_compute_text_features_for_row).apply(pd.Series)
    df = pd.concat([df, features], axis=1)
    print(f"Добавлены признаки: {TEXT_FEATURES}")
    return df


# ---------------------------------------------------------------------------
# Получение списка признаков
# ---------------------------------------------------------------------------


def get_all_feature_columns() -> list[str]:
    """Возвращает полный список признаков для обучения моделей.

    Включает аудио-признаки, тематические вероятности,
    текстовые признаки и метаданные.

    Returns
    -------
    list[str]
        Список названий столбцов-признаков.
    """
    return AUDIO_FEATURES + TOPIC_FEATURES + TEXT_FEATURES + META_FEATURES


# ---------------------------------------------------------------------------
# Разбивка на выборки
# ---------------------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированный сплит на train/val/test по колонке genre.

    Порядок: сначала отделяется test, затем из оставшегося — val.
    Стратификация гарантирует, что пропорции жанров сохраняются
    во всех трёх выборках — это предотвращает утечку данных.

    Parameters
    ----------
    df:
        Очищенный датасет с признаками.
    test_size:
        Доля тестовой выборки (от всего датасета).
    val_size:
        Доля валидационной выборки (от всего датасета).
    random_state:
        Зерно генератора случайных чисел для воспроизводимости.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    # Отделяем test
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["genre"],
    )

    # Из оставшегося отделяем val
    # Пересчитываем долю val от train_val
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val["genre"],
    )

    print(
        f"Размеры выборок: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(всего: {len(df)})"
    )
    print(
        f"Доли: train={len(train)/len(df):.1%}, "
        f"val={len(val)/len(df):.1%}, "
        f"test={len(test)/len(df):.1%}"
    )
    return train, val, test
