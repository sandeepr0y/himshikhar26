from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
import numpy as np


def get_pipeline(transformer):
    mp = Pipeline(steps=[
        ("preprocessor", transformer),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        ),
    ])

    # Wrap the pipeline in a TransformedTargetRegressor
    # func=np.log1p transforms price to log(price + 1)
    # inverse_func=np.expm1 converts it back to real dollars automatically on predict()
    return TransformedTargetRegressor(
        regressor=mp,
        func=np.log1p,
        inverse_func=np.expm1
    )