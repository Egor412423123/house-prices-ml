import numpy as np
import pandas as pd

from src.model import (
    build_model_pipeline,
    get_boosting_model_specs,
    get_linear_model_specs,
    get_neural_model_specs,
    get_stacking_model_specs,
    get_tree_model_specs,
)


def test_linear_and_tree_registries_have_expected_current_keys():
    linear = get_linear_model_specs()
    tree = get_tree_model_specs()

    assert set(linear) == {"linear_regression", "ridge", "lasso", "elastic_net"}
    assert set(tree) == {"decision_tree", "random_forest", "extra_trees"}
    assert all(spec.description for spec in {**linear, **tree}.values())


def test_registries_return_fresh_estimators():
    first = get_linear_model_specs()
    second = get_linear_model_specs()

    assert first["ridge"].estimator is not second["ridge"].estimator


def test_boosting_registry_has_current_models():
    specs = get_boosting_model_specs()

    assert set(specs) == {
        "adaboost",
        "gradient_boosting",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    }
    assert all(spec.family == "boosting" for spec in specs.values())


def test_neural_registry_contains_current_mlp():
    specs = get_neural_model_specs()

    assert set(specs) == {"mlp"}
    assert specs["mlp"].family == "neural_network"
    assert specs["mlp"].requires_dense


def test_model_pipeline_fits_mixed_data_and_predicts_finite_values():
    frame = pd.DataFrame(
        {
            "Id": np.arange(1, 13),
            "MSSubClass": [20, 60] * 6,
            "MoSold": [1, 2, 3, 4, 5, 6] * 2,
            "LotArea": np.linspace(6_000, 12_000, 12),
            "Neighborhood": ["NAmes", "OldTown", "CollgCr"] * 4,
            "Alley": [None, "Grvl", None] * 4,
        }
    )
    target = pd.Series(np.linspace(11.0, 12.0, len(frame)))
    pipeline = build_model_pipeline(get_linear_model_specs()["ridge"], frame)

    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)

    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()


def test_dense_boosting_pipeline_fits_mixed_data():
    frame = pd.DataFrame(
        {
            "Id": np.arange(1, 41),
            "MSSubClass": [20, 60] * 20,
            "MoSold": list(range(1, 11)) * 4,
            "LotArea": np.linspace(6_000, 15_000, 40),
            "Neighborhood": ["NAmes", "OldTown", "CollgCr", "Somerst"] * 10,
            "Alley": [None, "Grvl"] * 20,
        }
    )
    target = pd.Series(np.linspace(10.8, 12.3, len(frame)))
    spec = get_boosting_model_specs()["hist_gradient_boosting"]
    pipeline = build_model_pipeline(spec, frame)

    pipeline.fit(frame, target)
    predictions = pipeline.predict(frame)

    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()


def test_stacking_registry_builds_from_current_model_names():
    frame = pd.DataFrame({"LotArea": [1, 2], "Neighborhood": ["A", "B"]})
    specs = get_stacking_model_specs(
        frame,
        combinations={"small": ["ridge", "decision_tree"]},
    )

    assert set(specs) == {"small"}
    assert specs["small"].family == "stacking"
