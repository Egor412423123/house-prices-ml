import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.validation import (
    cross_validate_regressor,
    evaluate_oof_ensembles,
    generate_oof_predictions,
    make_cv,
    residual_correlation,
    randomized_tune_regressor,
    split_development_holdout,
)
from src.model import get_linear_model_specs, get_tree_model_specs


def make_sample_data(n_rows=100):
    features = pd.DataFrame(
        {"area": np.arange(n_rows), "quality": np.arange(n_rows) % 10}
    )
    target = pd.Series(np.linspace(10.0, 13.0, n_rows), name="target_log")
    return features, target


def test_development_holdout_split_is_deterministic_and_disjoint():
    features, target = make_sample_data()

    first = split_development_holdout(features, target)
    second = split_development_holdout(features, target)
    x_dev, x_holdout, y_dev, y_holdout = first

    assert len(x_dev) == 80
    assert len(x_holdout) == 20
    assert set(x_dev.index).isdisjoint(x_holdout.index)
    assert set(x_dev.index) | set(x_holdout.index) == set(features.index)
    assert x_dev.index.equals(y_dev.index)
    assert x_holdout.index.equals(y_holdout.index)
    assert first[0].index.equals(second[0].index)
    assert first[1].index.equals(second[1].index)


def test_kfold_validation_covers_each_development_row_once():
    features, target = make_sample_data()
    cv = make_cv()
    validation_indices = []

    for train_indices, fold_indices in cv.split(features, target):
        assert set(train_indices).isdisjoint(fold_indices)
        validation_indices.extend(fold_indices)

    assert sorted(validation_indices) == list(range(len(features)))


def test_cross_validate_regressor_returns_positive_rmse_for_each_fold():
    features, target = make_sample_data()
    results = cross_validate_regressor(
        DummyRegressor(strategy="median"),
        features,
        target,
        n_jobs=1,
    )

    assert len(results) == 5
    assert results["validation_rmse"].gt(0).all()
    assert np.isfinite(results["validation_rmse"]).all()


def test_randomized_tuning_returns_ranked_positive_rmse():
    features, target = make_sample_data(40)
    search, results = randomized_tune_regressor(
        DummyRegressor(),
        features,
        target,
        param_distributions={"strategy": ["mean", "median"]},
        cv=make_cv(n_splits=2),
        n_iter=2,
        n_jobs=1,
    )

    assert search.best_params_["strategy"] in {"mean", "median"}
    assert list(results["rank_test_score"]) == sorted(results["rank_test_score"])
    assert results["cv_rmse_mean"].gt(0).all()


def test_oof_predictions_cover_every_row_and_ensembles_are_finite():
    rows = 30
    frame = pd.DataFrame(
        {
            "Id": np.arange(1, rows + 1),
            "MSSubClass": [20, 60, 70] * 10,
            "LotArea": np.linspace(5_000, 15_000, rows),
            "Neighborhood": ["NAmes", "OldTown", "CollgCr"] * 10,
        }
    )
    target = pd.Series(np.linspace(10.5, 12.5, rows))
    specs = {
        "ridge": get_linear_model_specs()["ridge"],
        "decision_tree": get_tree_model_specs()["decision_tree"],
    }

    oof, fold_metrics = generate_oof_predictions(
        specs, frame, target, cv=make_cv(n_splits=3)
    )
    predictions, summary, weights = evaluate_oof_ensembles(oof, list(specs))

    assert set(oof["fold"]) == {1, 2, 3}
    assert not oof[list(specs)].isna().any().any()
    assert len(fold_metrics) == 6
    assert np.isfinite(predictions["UniformAverage"]).all()
    assert np.isclose(weights["optimized_weight"].sum(), 1.0)
    assert "RidgeStacking" in set(summary["model"])
    assert residual_correlation(oof, list(specs)).shape == (2, 2)

