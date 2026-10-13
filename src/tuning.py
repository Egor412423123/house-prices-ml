
from __future__ import annotations

import numpy as np
import optuna

from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_validate

from .features import build_preprocessor


def to_dense(X):
    """Convert sparse matrix to dense when required."""
    return X.toarray() if sparse.issparse(X) else X


def objective(trial, name, spec, X_train, y_train, cv):
    """
    Optimize boosting hyperparameters using cross-validation.

    Returns:
        Mean validation RMSE across CV folds.
    """

    # 1. AdaBoost
    if name == "adaboost":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 500, step=50
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 1.0, log=True
            ),
            "loss": trial.suggest_categorical(
                "loss", ["linear", "square", "exponential"]
            ),
        }

    # 2. Gradient Boosting
    elif name == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 100, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 6
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 15
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
        }

    # 3. Histogram Gradient Boosting
    elif name == "hist_gradient_boosting":
        params = {
            "max_iter": trial.suggest_int(
                "max_iter", 100, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", 7, 63
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 12
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 5, 40
            ),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", 1e-4, 10.0, log=True
            ),
        }

    # 4. XGBoost
    elif name == "xgboost":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 2, 8
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 20.0, log=True
            ),
        }

    # 5. LightGBM
    elif name == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", 200, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 7, 63
            ),
            "max_depth": trial.suggest_int(
                "max_depth", 3, 12
            ),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 40
            ),
            "subsample": trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            "subsample_freq": trial.suggest_int(
                "subsample_freq", 1, 5
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 0.01, 20.0, log=True
            ),
        }

    # 6. CatBoost
    elif name == "catboost":
        params = {
            "iterations": trial.suggest_int(
                "iterations", 200, 800, step=100
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "depth": trial.suggest_int(
                "depth", 3, 8
            ),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1.0, 20.0, log=True
            ),
            "random_strength": trial.suggest_float(
                "random_strength", 0.01, 5.0, log=True
            ),
        }

    else:
        raise ValueError(f"Unknown model: {name}")

    # Create an independent estimator
    estimator = clone(spec.estimator)
    estimator.set_params(**params)

    # Avoid nested parallelism
    if name in ("xgboost", "lightgbm"):
        estimator.set_params(n_jobs=1)

    elif name == "catboost":
        estimator.set_params(thread_count=1)

    # Create preprocessing pipeline
    steps = [
        (
            "preprocessing",
            build_preprocessor(
                X_train,
                scale_numeric=spec.scale_numeric
            )
        )
    ]

    # Convert sparse to dense if required
    if spec.requires_dense:
        steps.append(
            (
                "to_dense",
                FunctionTransformer(
                    to_dense,
                    accept_sparse=True
                )
            )
        )

    # Add estimator
    steps.append(("model", estimator))

    pipeline = Pipeline(steps)

    # Evaluate with cross-validation
    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        error_score="raise"
    )

    return -np.mean(scores)


def objective_stacking(trial, stacking_spec, X_train, y_train, cv):

    # Выбираем тип метамодели
    meta_model = trial.suggest_categorical(
        "meta_model", ["ridge", "elastic_net"]
    )

    if meta_model == "ridge":
        estimator = Ridge(
            alpha=trial.suggest_float(
                "ridge_alpha", 1e-3, 100, log=True
            )
        )

    else:
        estimator = ElasticNet(
            alpha=trial.suggest_float(
                "elastic_alpha", 1e-5, 1.0, log=True
            ),
            l1_ratio=trial.suggest_float(
                "elastic_l1_ratio", 0.1, 1.0
            ),
            max_iter=10000
        )

    # Берём готовую конфигурацию Stacking D
    stacking = clone(stacking_spec.estimator)

    # Меняем только метамодель
    stacking.set_params(final_estimator=estimator)

    # Внешняя кросс-валидация
    scores = cross_validate(
        stacking,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise"
    )

    return -scores["test_score"].mean()

def objective_mlp_optimizer(trial, pipeline, X_train, y_train, cv):

    solver = trial.suggest_categorical(
        "solver", ["adam", "sgd", "lbfgs"]
    )

    model = clone(pipeline)

    params = {
        "model__solver": solver,
        "model__hidden_layer_sizes": (32, 16),
        "model__alpha": 0.01,
        "model__early_stopping": False,
        "model__random_state": 42,
        "model__max_iter": 1500,
    }

    if solver == "adam":
        params["model__learning_rate_init"] = trial.suggest_float(
            "adam_lr", 1e-4, 1e-2, log=True
        )
        params["model__batch_size"] = trial.suggest_categorical(
            "adam_batch_size", [16, 32, 64]
        )

    elif solver == "sgd":
        params["model__learning_rate_init"] = trial.suggest_float(
            "sgd_lr", 1e-4, 1e-2, log=True
        )
        params["model__momentum"] = trial.suggest_float(
            "sgd_momentum", 0.5, 0.99
        )
        params["model__batch_size"] = trial.suggest_categorical(
            "sgd_batch_size", [16, 32, 64]
        )
        params["model__learning_rate"] = "adaptive"

    elif solver == "lbfgs":
        params["model__max_fun"] = 20000

    model.set_params(**params)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise"
    )

    return -scores["test_score"].mean()


def objective_mlp_architecture(trial, pipeline, X_train, y_train, cv):

    n_layers = trial.suggest_int("n_layers", 1, 3)

    hidden_layers = tuple(
        trial.suggest_categorical(
            f"n_units_layer_{i}", [16, 32, 64, 128]
        )
        for i in range(n_layers)
    )

    alpha = trial.suggest_float(
        "alpha", 1e-5, 1e-1, log=True
    )

    model = clone(pipeline)

    model.set_params(
        model__solver="sgd",
        model__learning_rate="adaptive",
        model__learning_rate_init=0.008706,
        model__momentum=0.907897,
        model__batch_size=16,
        model__hidden_layer_sizes=hidden_layers,
        model__alpha=alpha,
        model__early_stopping=False,
        model__max_iter=1500,
        model__random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        error_score="raise"
    )

    return -scores.mean()