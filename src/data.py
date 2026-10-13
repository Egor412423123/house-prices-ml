from pathlib import Path

import pandas as pd


def load_competition_data(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle train and test tables from ``raw_dir``."""
    raw_dir = Path(raw_dir)
    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    missing = [path.name for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Не найдены файлы: {', '.join(missing)}. Поместите их в {raw_dir}."
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)

