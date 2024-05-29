"""
This is a boilerplate pipeline 'data_prediction'
generated using Kedro 0.19.6
"""


import pandas as pd
import mlflow


def train_model(run_id, data: pd.DataFrame) -> pd.DataFrame:
    model = mlflow.sklearn.load_model(f"mlruns:/328428339397235956/76d92bf71a064058abc52d9097337f78")
    predict_data = model.predict(data)
    return predict_data

