
"""Воспроизведение финальных моделей House Prices.

Запуск:
    python run_all.py
    python run_all.py --with-tests

Использует сохранённые гиперпараметры без Optuna и GridSearchCV.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.model import (
    build_model_pipeline,
    get_boosting_model_specs,
    get_stacking_model_specs,
)


# ============================================================
# 1. Пути проекта
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

BOOSTING_CONFIG = REPORTS_DIR / "boosting_best_params.json"
STACKING_CONFIG = REPORTS_DIR / "stacking_best_params.json"


# ============================================================
# 2. Вспомогательные функции
# ============================================================

def load_json(path: Path) -> dict:
    """Загрузить сохранённую конфигурацию модели."""
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def check_required_files() -> None:
    """Проверить наличие данных и конфигураций."""
    required = [
        TRAIN_PATH,
        TEST_PATH,
        BOOSTING_CONFIG,
        STACKING_CONFIG,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        names = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Не найдены обязательные файлы:\n{names}"
        )


def save_submission(
    model,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    filename: str,
) -> None:
    """Получить предсказания и создать Kaggle submission."""

    # Модели обучены на log1p(SalePrice)
    predictions_log = model.predict(X_test)

    # Восстанавливаем исходный масштаб цен
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    submission = pd.DataFrame({
        "Id": test_ids.to_numpy(),
        "SalePrice": predictions,
    })

    # Проверки перед сохранением
    assert submission.shape == (len(X_test), 2)
    assert list(submission.columns) == ["Id", "SalePrice"]
    assert submission["Id"].is_unique
    assert np.isfinite(submission["SalePrice"]).all()
    assert (submission["SalePrice"] >= 0).all()

    output_path = PROJECT_ROOT / filename
    submission.to_csv(output_path, index=False)

    print(f"Сохранено: {output_path}")
    print(f"Количество строк: {len(submission)}")


# ============================================================
# 3. Финальный CatBoost
# ============================================================

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
) -> None:

    print("\n[1/2] Обучение CatBoost...")

    config = load_json(BOOSTING_CONFIG)

    spec = get_boosting_model_specs()["catboost"]

    # Полный Pipeline: preprocessing -> dense -> CatBoost
    model = build_model_pipeline(spec, X_train)

    # Применяем лучшие параметры Optuna
    best_params = config["catboost"]

    model.set_params(**{
        f"model__{key}": value
        for key, value in best_params.items()
    })

    model.fit(X_train, y_train)

    save_submission(
        model,
        X_test,
        test_ids,
        "submission_catboost.csv",
    )


# ============================================================
# 4. Финальный Stacking
# ============================================================

def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
) -> None:

    print("\n[2/2] Обучение Stacking...")

    config = load_json(STACKING_CONFIG)

    stacking_name = config["model"]
    base_models = config["base_models"]

    # Создаём исходный StackingRegressor
    specs = get_stacking_model_specs(
        X_train,
        combinations={
            stacking_name: base_models
        },
    )

    model = clone(specs[stacking_name].estimator)

    # Лучший alpha для Ridge из Optuna
    ridge_alpha = config["final_estimator"]["params"]["alpha"]

    model.set_params(
        final_estimator__alpha=ridge_alpha
    )

    print("Базовые модели:", ", ".join(base_models))
    print("Ridge alpha:", ridge_alpha)

    model.fit(X_train, y_train)

    save_submission(
        model,
        X_test,
        test_ids,
        "submission_stacking.csv",
    )


# ============================================================
# 5. Необязательное тестирование
# ============================================================

def run_tests() -> None:
    """Запустить pytest по желанию пользователя."""

    print("\nЗапуск pytest...")

    subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v"],
        cwd=PROJECT_ROOT,
        check=True,
    )


# ============================================================
# 6. Точка входа
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Обучить финальные модели House Prices."
    )

    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Дополнительно запустить pytest.",
    )

    args = parser.parse_args()

    started = time.perf_counter()

    try:
        # Проверяем необходимые файлы
        check_required_files()

        print("=" * 60)
        print("HOUSE PRICES — FINAL PIPELINE")
        print("=" * 60)

        # Загружаем исходные данные
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)

        X_train = train.drop(columns=["SalePrice"])
        y_train = np.log1p(train["SalePrice"])

        X_test = test.copy()
        test_ids = test["Id"].copy()

        print(f"Train: {X_train.shape}")
        print(f"Test:  {X_test.shape}")

        # Обучаем CatBoost
        train_catboost(
            X_train,
            y_train,
            X_test,
            test_ids,
        )

        # Обучаем Stacking
        train_stacking(
            X_train,
            y_train,
            X_test,
            test_ids,
        )

        # Необязательные тесты
        if args.with_tests:
            run_tests()

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        AssertionError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"\nОшибка: {error}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started

    print("\n" + "=" * 60)
    print("PIPELINE УСПЕШНО ЗАВЕРШЁН")
    print("=" * 60)

    print("Созданы файлы:")
    print("- submission_catboost.csv")
    print("- submission_stacking.csv")
    print(f"\nВремя выполнения: {elapsed / 60:.1f} мин.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
