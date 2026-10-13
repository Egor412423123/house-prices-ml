# House Prices — Advanced Regression Techniques

Учебный ML-проект по соревнованию Kaggle: предсказание цены продажи дома (`SalePrice`) по табличным признакам.

## Структура

```text
house-prices-project/
├── data/
│   ├── raw/                 # train.csv, test.csv, sample_submission.csv
│   └── processed/           # подготовленные данные
├── docs/                    # заметки и описание результатов
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   └── train.py
├── tests/
│   └── test_features.py
├── .gitignore
└── requirements.txt
```

## Быстрый старт

1. Данные соревнования находятся в `data/raw/`.
2. Создайте окружение и установите зависимости:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

3. Запустите Jupyter и откройте `notebooks/01_exploration.ipynb`.

## Запуск всего pipeline

Чтобы заново выполнить проект с этапа предобработки, запустите из корня
проекта:

```bash
python run_all.py
```

Команда последовательно выполнит `02_preprocessing.ipynb` и
`03_modeling.ipynb`. Результаты выполнения сохранятся непосредственно в
ноутбуках. При ошибке pipeline остановится на проблемной ячейке.

Для дополнительного запуска тестов после ноутбуков:

```bash
python run_all.py --with-tests
```

## План работы

- Изучить целевую переменную, пропуски, типы признаков и выбросы.
- Собрать воспроизводимый конвейер предобработки.
- Получить базовую модель и оценивать её по RMSE для `log1p(SalePrice)`.
- Сравнить несколько моделей и сформировать `submission.csv`.
