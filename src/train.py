import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def train_baseline(train_frame, random_state: int = 42):
    """Fit a baseline model and return it together with validation RMSLE."""
    x = train_frame.drop(columns=["SalePrice"])
    y = np.log1p(train_frame["SalePrice"])
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(x_train)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    score = root_mean_squared_error(y_valid, model.predict(x_valid))
    return model, score

