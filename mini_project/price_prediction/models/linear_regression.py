from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


def get_pipeline(transformer):
    return Pipeline(steps=[
        ('preprocessor', transformer),
        ('regressor', LinearRegression())
    ])