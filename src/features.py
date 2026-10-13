"""Feature preparation for the House Prices dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy import sparse

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"

# These columns contain numeric codes, but their values are labels rather than quantities.
CATEGORICAL_CODE_COLUMNS = ("MSSubClass", "MoSold")

# In the official data description, NA means that the physical feature is absent.
ABSENCE_CATEGORICAL_COLUMNS = (
    "Alley",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
)


def prepare_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns and correct categorical code dtypes."""
    prepared = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore").copy()

    for column in CATEGORICAL_CODE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype(str)

    return prepared


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names."""
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = frame.select_dtypes(exclude="number").columns.tolist()
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    """Build leakage-safe preprocessing fitted only on training observations.

    The returned object accepts the original feature table. It removes ``Id``,
    converts numeric category codes to strings, imputes missing values and
    one-hot encodes categorical columns while tolerating unseen categories.
    """
    prepared = prepare_feature_frame(frame)
    numeric, categorical = split_feature_types(prepared)
    absence_categorical = [
        column for column in ABSENCE_CATEGORICAL_COLUMNS if column in categorical
    ]
    regular_categorical = [
        column for column in categorical if column not in absence_categorical
    ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    columns = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "absence",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="None"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                absence_categorical,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                regular_categorical,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "prepare",
                FunctionTransformer(prepare_feature_frame, validate=False),
            ),
            ("columns", columns),
        ]
    )


def get_feature_names(fitted_preprocessor: Pipeline) -> list[str]:
    """Return output feature names from a fitted preprocessing pipeline."""
    return fitted_preprocessor.named_steps["columns"].get_feature_names_out().tolist()


def _to_dense(matrix):
    """Convert sparse preprocessing output for estimators requiring dense arrays."""
    return matrix.toarray() if sparse.issparse(matrix) else matrix
