from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def get_pipeline(transformer):
    return Pipeline(steps=[
        ('preprocessor', transformer),
        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
    ])