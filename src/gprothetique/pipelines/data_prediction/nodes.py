"""
This is a boilerplate pipeline 'data_prediction'
generated using Kedro 0.19.6
"""


import pandas as pd
import mlflow


def train_model(run_id, data: pd.DataFrame) -> pd.DataFrame:
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    print("------------------------------")
    print(data)

    after_cols = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz", "after_exam_4000_Hz", "after_exam_8000_Hz"]

    predict_data = pd.DataFrame(model.predict(data), columns=after_cols)
    return predict_data

