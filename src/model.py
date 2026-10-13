from __future__ import annotations

from dataclasses import dataclass

from scipy import sparse
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from .features import build_preprocessor, _to_dense
from .validation import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    """A model together with preprocessing and reporting metadata."""

    estimator: object
    family: str
    scale_numeric: bool
    description: str
    requires_dense: bool = False
def get_linear_model_specs(
    random_state: int = RANDOM_STATE,
) -> dict[str, ModelSpec]:
    
    return {
       
        "linear_regression": ModelSpec(
            estimator=LinearRegression(),
            family="linear",
            scale_numeric=True,
            description="Ordinary least squares regression"
        ),

        "ridge": ModelSpec(
            estimator=Ridge(alpha=1.0),
            family="linear",
            scale_numeric=True,
            description="Linear regression with L2 regularization"
        ),

        "lasso": ModelSpec(
            estimator=Lasso(alpha=1.0, max_iter=10000),
            family="linear",
            scale_numeric=True,
            description="Linear regression with L1 regularization"
        ),
        
        "elastic_net": ModelSpec(
            estimator=ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42
            ),
            family="linear",
            scale_numeric=True,
            description="Linear regression with combined L1 and L2 regularization"
        )
    }
    

def get_tree_model_specs(
    random_state: int = RANDOM_STATE,
) -> dict[str, ModelSpec]:

    return {
        "decision_tree": ModelSpec(
            estimator=DecisionTreeRegressor(
                random_state=random_state
            ),
            family="tree",
            scale_numeric=False,
            description="Single decision tree"
        ),

        "random_forest": ModelSpec(
            estimator=RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=1
            ),
            family="bagging",
            scale_numeric=False,
            description="Random forest"
        ),

        "extra_trees": ModelSpec(
            estimator=ExtraTreesRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=1
            ),
            family="bagging",
            scale_numeric=False,
            description="Extremely randomized trees"
        )
    }
def get_boosting_model_specs(
    random_state: int = RANDOM_STATE,
) -> dict[str, ModelSpec]:
    
    try:
        from catboost import CatBoostRegressor
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError(
            "Install optional boosters with: "
            "python -m pip install -r requirements-boosting.txt"
        ) from error

    return {
        "adaboost": ModelSpec(
            estimator=AdaBoostRegressor(
                n_estimators=100,
                random_state=random_state
            ),
            family="boosting",
            scale_numeric=False,
            description="Adaptive Boosting"
        ),

        "gradient_boosting": ModelSpec(
            estimator=GradientBoostingRegressor(
                n_estimators=200,
                random_state=random_state
            ),
            family="boosting",
            scale_numeric=False,
            description="Classic Gradient Boosting"
        ),

        "hist_gradient_boosting": ModelSpec(
            estimator=HistGradientBoostingRegressor(
                max_iter=200,
                random_state=random_state
            ),
            family="boosting",
            scale_numeric=False,
            description="Histogram Gradient Boosting",
            requires_dense=True
        ),

        "xgboost": ModelSpec(
            estimator=XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                random_state=random_state,
                n_jobs=1
            ),
            family="boosting",
            scale_numeric=False,
            description="XGBoost"
        ),

        "lightgbm": ModelSpec(
            estimator=LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                random_state=random_state,
                n_jobs=1,
                verbosity=-1
            ),
            family="boosting",
            scale_numeric=False,
            description="LightGBM"
        ),

        "catboost": ModelSpec(
            estimator=CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=random_state,
                verbose=False,
                thread_count=1,
                allow_writing_files=False
            ),
            family="boosting",
            scale_numeric=False,
            description="CatBoost",
            requires_dense=True
        )
    }

def get_stacking_model_specs(
    X_train,
    combinations=None,
    random_state=RANDOM_STATE
):

    if combinations is None:
        combinations = {
            "stacking_a": ["elastic_net", "lightgbm"],
            "stacking_b": ["lightgbm", "catboost"],
            "stacking_c": ["elastic_net", "lightgbm", "catboost"],
            "stacking_d": [
                "elastic_net",
                "random_forest",
                "lightgbm",
                "catboost"
            ]
        }

    # Получаем спецификации моделей
    linear_specs = get_linear_model_specs(random_state)
    tree_specs = get_tree_model_specs(random_state)
    boosting_specs = get_boosting_model_specs(random_state)

    all_specs = {
        **linear_specs,
        **tree_specs,
        **boosting_specs
    }

    results = {}

    for stacking_name, model_names in combinations.items():

        estimators = []

        for model_name in model_names:

            spec = all_specs[model_name]

            steps = [
                (
                    "preprocessing",
                    build_preprocessor(
                        X_train,
                        scale_numeric=spec.scale_numeric
                    )
                )
            ]

            if spec.requires_dense:
                steps.append(
                    (
                        "to_dense",
                        FunctionTransformer(
                            _to_dense,
                            accept_sparse=True
                        )
                    )
                )

            steps.append(
                ("model", clone(spec.estimator))
            )

            estimators.append(
                (model_name, Pipeline(steps))
            )

        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=1,
            passthrough=False
        )

        results[stacking_name] = ModelSpec(
            estimator=stacking,
            family="stacking",
            scale_numeric=False,
            description=f"Stacking: {', '.join(model_names)}"
        )

    return results


def get_neural_model_specs(
    random_state: int = RANDOM_STATE,
) -> dict[str, ModelSpec]:

    return {
        "mlp": ModelSpec(
            estimator=MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                batch_size=32,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=random_state
            ),
            family="neural_network",
            scale_numeric=True,
            description="MLP with two hidden layers",
            requires_dense=True
        )
    }


def build_model_pipeline(spec: ModelSpec, frame) -> Pipeline:
    """Build a preprocessing and estimator pipeline for one model spec."""
    steps = [
        (
            "preprocessing",
            build_preprocessor(frame, scale_numeric=spec.scale_numeric),
        )
    ]
    if spec.requires_dense:
        steps.append(
            (
                "to_dense",
                FunctionTransformer(_to_dense, accept_sparse=True),
            )
        )
    steps.append(("model", clone(spec.estimator)))
    return Pipeline(steps)
