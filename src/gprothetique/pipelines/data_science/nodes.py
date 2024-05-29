import logging
from typing import Dict, Tuple
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()


def train_model(data_train: pd.DataFrame) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    target = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz",
              "after_exam_4000_Hz", "after_exam_8000_Hz"]
    X_train = data_train[target]
    y_train = data_train.drop(target, axis=1)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    mlflow.sklearn.log_model(regressor, "regressor")

    print('tracking uri:', mlflow.get_tracking_uri())
    print('tracking uri:', mlflow.get_artifact_uri())
    return regressor


def evaluate_model(regressor: LinearRegression, test_data: pd.DataFrame) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    target = ["after_exam_125_Hz", "after_exam_250_Hz", "after_exam_500_Hz", "after_exam_1000_Hz", "after_exam_2000_Hz",
              "after_exam_4000_Hz", "after_exam_8000_Hz"]
    X_test = test_data[target]
    y_test = test_data.drop(target, axis=1)

    y_pred = regressor.predict(X_test)

    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae}
