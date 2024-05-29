"""
This is a boilerplate pipeline 'data_prediction'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_model,
            inputs=["test_data", "params:run_id"],
            outputs="data_predict",
            name="prediction_node",
        ),
    ])