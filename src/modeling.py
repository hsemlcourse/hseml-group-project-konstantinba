"""
Вспомогательные функции для обучения и оценки моделей.
Используются в ноутбуке 03_experiments.ipynb.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate(model, X_train, y_train, X_val, y_val, name: str) -> dict:
    """Обучает модель и возвращает метрики на val-выборке.

    Parameters
    ----------
    model:
        Классификатор или Pipeline с методами fit/predict.
    X_train, y_train:
        Обучающая выборка.
    X_val, y_val:
        Валидационная выборка.
    name:
        Название модели для отчёта.

    Returns
    -------
    dict
        Словарь с ключами: Модель, Macro F1 (val), Accuracy (val).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"{name}: Macro F1 = {macro_f1:.4f}, Accuracy = {accuracy:.4f}")
    return {
        "Модель": name,
        "Macro F1 (val)": round(macro_f1, 4),
        "Accuracy (val)": round(accuracy, 4),
    }


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """Строит тепловую карту матрицы ошибок.

    Parameters
    ----------
    y_true:
        Истинные метки.
    y_pred:
        Предсказанные метки.
    labels:
        Порядок классов для осей.
    title:
        Заголовок графика.
    save_path:
        Путь для сохранения PNG. Если None — только показывает.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Предсказанный жанр")
    ax.set_ylabel("Истинный жанр")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Матрица ошибок сохранена: {save_path}")
    plt.show()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str,
    save_path: Optional[str] = None,
    top_n: int = 20,
) -> None:
    """Строит горизонтальный bar-chart важности признаков.

    Parameters
    ----------
    importances:
        Массив важностей (feature_importances_ модели).
    feature_names:
        Названия признаков (в том же порядке).
    title:
        Заголовок графика.
    save_path:
        Путь для сохранения PNG. Если None — только показывает.
    top_n:
        Сколько топ-признаков показывать.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(top_n), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Важность признака")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График важности сохранён: {save_path}")
    plt.show()


def build_experiment_row(
    name: str,
    hypothesis: str,
    method: str,
    macro_f1_val: float,
    accuracy_val: float,
    macro_f1_test: Optional[float] = None,
    accuracy_test: Optional[float] = None,
    note: str = "",
) -> dict:
    """Формирует строку для итоговой таблицы экспериментов.

    Parameters
    ----------
    name:
        Название эксперимента.
    hypothesis:
        Гипотеза, которую проверяет эксперимент.
    method:
        Метод/алгоритм.
    macro_f1_val, accuracy_val:
        Метрики на валидационной выборке.
    macro_f1_test, accuracy_test:
        Метрики на тестовой выборке (опционально).
    note:
        Краткий вывод по эксперименту.

    Returns
    -------
    dict
        Строка таблицы экспериментов.
    """
    return {
        "Эксперимент": name,
        "Гипотеза": hypothesis,
        "Метод": method,
        "Macro F1 (val)": round(macro_f1_val, 4),
        "Accuracy (val)": round(accuracy_val, 4),
        "Macro F1 (test)": round(macro_f1_test, 4)
        if macro_f1_test is not None
        else "-",
        "Accuracy (test)": round(accuracy_test, 4)
        if accuracy_test is not None
        else "-",
        "Вывод": note,
    }


def print_results_table(
    results: list[dict], sort_by: str = "Macro F1 (val)"
) -> pd.DataFrame:
    """Выводит сводную таблицу результатов экспериментов.

    Parameters
    ----------
    results:
        Список словарей-строк (из evaluate или build_experiment_row).
    sort_by:
        Столбец для сортировки (по убыванию).

    Returns
    -------
    pd.DataFrame
        Отсортированная таблица результатов.
    """
    df = (
        pd.DataFrame(results)
        .sort_values(sort_by, ascending=False)
        .reset_index(drop=True)
    )
    print("=" * 70)
    print(f"Сводная таблица результатов (сортировка по {sort_by}):")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df
