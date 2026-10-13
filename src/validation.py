"""Reproducible validation utilities for regression experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.model_selection import cross_val_predict


RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
CV_FOLDS = 5


def split_development_holdout(
    features: pd.DataFrame,
    target_log: pd.Series,
    *,
    test_size: float = HOLDOUT_SIZE,
    random_state: int = RANDOM_STATE,
    n_target_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible 80/20 split stratified by target quantiles.

    Stratification uses only the target to balance cheap and expensive houses
    across the development and holdout sets. Original indices are preserved.
    """
    if len(features) != len(target_log):
        raise ValueError("features and target_log must contain the same number of rows")
    if not features.index.equals(target_log.index):
        raise ValueError("features and target_log indices must match")

    target_bins = pd.qcut(
        target_log,
        q=n_target_bins,
        labels=False,
        duplicates="drop",
    )

    return train_test_split(
        features,
        target_log,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=target_bins,
    )


def make_cv(
    *,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> KFold:
    """Return the shared shuffled K-fold strategy used by all models."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def cross_validate_regressor(
    estimator,
    features: pd.DataFrame,
    target_log: pd.Series,
    *,
    cv=None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Evaluate an estimator with log-target RMSE and return fold-level results."""
    cv = cv or make_cv()
    scores = cross_validate(
        estimator,
        features,
        target_log,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
        n_jobs=n_jobs,
        error_score="raise",
    )

    return pd.DataFrame(
        {
            "fold": np.arange(1, len(scores["test_score"]) + 1),
            "train_rmse": -scores["train_score"],
            "validation_rmse": -scores["test_score"],
            "fit_time_seconds": scores["fit_time"],
            "score_time_seconds": scores["score_time"],
        }
    )


def randomized_tune_regressor(
    estimator,
    features,
    target_log,
    *,
    param_distributions,
    cv,
    n_iter=12,
    random_state=RANDOM_STATE,
    n_jobs=1,
):
    """Tune a complete pipeline using only development cross-validation."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        refit=True,
        return_train_score=True,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score="raise",
    )
    search.fit(features, target_log)

    raw_results = pd.DataFrame(search.cv_results_)
    result_columns = [
        column
        for column in raw_results.columns
        if column.startswith("param_")
    ]
    results = raw_results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "mean_fit_time",
            "params",
            *result_columns,
        ]
    ].copy()
    results["cv_rmse_mean"] = -results.pop("mean_test_score")
    results["cv_rmse_std"] = results.pop("std_test_score")
    results["train_rmse_mean"] = -results.pop("mean_train_score")
    results = results.sort_values("rank_test_score").reset_index(drop=True)
    return search, results


def generate_oof_predictions(model_specs, features, target_log, *, cv):
    """Fit every model on each training fold and return aligned OOF predictions."""
    import time

    from .model import build_model_pipeline

    if len(features) != len(target_log):
        raise ValueError("features and target_log must contain the same number of rows")
    if not model_specs:
        raise ValueError("model_specs must contain at least one model")

    n_rows = len(features)
    oof = pd.DataFrame(
        {
            "row_position": np.arange(n_rows),
            "source_index": np.asarray(features.index),
            "fold": np.zeros(n_rows, dtype=int),
            "y_true": np.asarray(target_log, dtype=float),
        }
    )
    fold_metrics = []
    first_model_name = next(iter(model_specs))

    for model_name, spec in model_specs.items():
        predictions = np.full(n_rows, np.nan, dtype=float)
        for fold, (train_positions, validation_positions) in enumerate(
            cv.split(features, target_log), start=1
        ):
            pipeline = build_model_pipeline(spec, features.iloc[train_positions])
            started_at = time.perf_counter()
            pipeline.fit(
                features.iloc[train_positions],
                target_log.iloc[train_positions],
            )
            fold_predictions = pipeline.predict(features.iloc[validation_positions])
            predictions[validation_positions] = fold_predictions

            existing_folds = oof.loc[validation_positions, "fold"]
            if model_name == first_model_name:
                oof.loc[validation_positions, "fold"] = fold
            elif not np.all(existing_folds.to_numpy() == fold):
                raise RuntimeError("All models must use identical folds")

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "validation_rmse": root_mean_squared_error(
                        target_log.iloc[validation_positions], fold_predictions
                    ),
                    "fit_predict_seconds": time.perf_counter() - started_at,
                }
            )

        if np.isnan(predictions).any():
            raise RuntimeError(f"Missing OOF predictions for {model_name}")
        oof[model_name] = predictions

    return oof, pd.DataFrame(fold_metrics)


def inverse_rmse_weights(oof_frame, model_columns):
    """Create transparent non-negative weights from individual OOF RMSE values."""
    scores = np.array(
        [
            root_mean_squared_error(oof_frame["y_true"], oof_frame[column])
            for column in model_columns
        ]
    )
    inverse_scores = 1.0 / scores
    return inverse_scores / inverse_scores.sum()


def optimize_blend_weights(oof_frame, model_columns):
    """Find non-negative weights summing to one on the OOF meta-dataset."""
    prediction_matrix = oof_frame[model_columns].to_numpy(dtype=float)
    target = oof_frame["y_true"].to_numpy(dtype=float)
    initial_weights = np.full(len(model_columns), 1.0 / len(model_columns))
    result = minimize(
        lambda weights: root_mean_squared_error(target, prediction_matrix @ weights),
        x0=initial_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(model_columns),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimization failed: {result.message}")
    return result.x


def evaluate_oof_ensembles(oof_frame, model_columns, *, random_state=RANDOM_STATE):
    """Add averaging, weighted blending and exploratory stacking predictions."""
    result = oof_frame.copy()
    prediction_matrix = result[model_columns].to_numpy(dtype=float)
    target = result["y_true"].to_numpy(dtype=float)

    uniform_weights = np.full(len(model_columns), 1.0 / len(model_columns))
    inverse_weights = inverse_rmse_weights(result, model_columns)
    optimized_weights = optimize_blend_weights(result, model_columns)
    result["UniformAverage"] = prediction_matrix @ uniform_weights
    result["InverseRMSEBlend"] = prediction_matrix @ inverse_weights
    result["OptimizedBlend"] = prediction_matrix @ optimized_weights

    meta_cv = make_cv(n_splits=5, random_state=random_state + 1)
    result["RidgeStacking"] = cross_val_predict(
        RidgeCV(alphas=np.logspace(-4, 3, 30)),
        prediction_matrix,
        target,
        cv=meta_cv,
        n_jobs=1,
    )

    ensemble_names = [
        "UniformAverage",
        "InverseRMSEBlend",
        "OptimizedBlend",
        "RidgeStacking",
    ]
    summary = pd.DataFrame(
        {
            "model": [*model_columns, *ensemble_names],
            "type": ["base_model"] * len(model_columns)
            + ["ensemble"] * len(ensemble_names),
            "oof_rmse": [
                root_mean_squared_error(target, result[column])
                for column in [*model_columns, *ensemble_names]
            ],
            "evaluation_note": ["outer_oof"] * len(model_columns)
            + [
                "honest_fixed_rule_oof",
                "weights_derived_from_same_oof",
                "weights_optimized_on_same_oof",
                "meta_model_on_oof_features_not_nested",
            ],
        }
    ).sort_values("oof_rmse")
    weights = pd.DataFrame(
        {
            "model": model_columns,
            "uniform_weight": uniform_weights,
            "inverse_rmse_weight": inverse_weights,
            "optimized_weight": optimized_weights,
        }
    )
    return result, summary, weights


def residual_correlation(oof_frame, model_columns):
    """Return correlations between OOF residuals of the base models."""
    residuals = pd.DataFrame(
        {
            column: oof_frame["y_true"] - oof_frame[column]
            for column in model_columns
        }
    )
    return residuals.corr()

