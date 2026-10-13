import numpy as np
import pandas as pd

from src.features import (
    build_preprocessor,
    get_feature_names,
    prepare_feature_frame,
    split_feature_types,
)


def test_split_feature_types():
    frame = pd.DataFrame({"area": [10, 20], "style": ["A", "B"]})

    numeric, categorical = split_feature_types(frame)

    assert numeric == ["area"]
    assert categorical == ["style"]


def test_prepare_feature_frame_drops_non_features_and_casts_codes():
    frame = pd.DataFrame(
        {
            "Id": [1],
            "SalePrice": [100_000],
            "MSSubClass": [20],
            "MoSold": [7],
            "LotArea": [8_000],
        }
    )

    prepared = prepare_feature_frame(frame)

    assert "Id" not in prepared
    assert "SalePrice" not in prepared
    assert prepared["MSSubClass"].dtype == object
    assert prepared["MoSold"].dtype == object


def test_preprocessor_handles_missing_and_unknown_categories():
    train = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "SalePrice": [100_000, 120_000, 140_000],
            "MSSubClass": [20, 60, 20],
            "MoSold": [5, 6, 7],
            "LotArea": [8_000.0, np.nan, 9_500.0],
            "Alley": [np.nan, "Grvl", np.nan],
            "Neighborhood": ["NAmes", "OldTown", "NAmes"],
        }
    )
    test = pd.DataFrame(
        {
            "Id": [4],
            "MSSubClass": [70],
            "MoSold": [8],
            "LotArea": [np.nan],
            "Alley": [np.nan],
            "Neighborhood": ["NewArea"],
        }
    )

    preprocessor = build_preprocessor(train)
    train_matrix = preprocessor.fit_transform(train)
    test_matrix = preprocessor.transform(test)
    names = get_feature_names(preprocessor)

    assert train_matrix.shape[0] == 3
    assert test_matrix.shape == (1, train_matrix.shape[1])
    assert not np.isnan(train_matrix.data).any()
    assert not np.isnan(test_matrix.data).any()
    assert "absence__Alley_None" in names

