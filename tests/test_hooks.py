import pytest
import mlflow
from src.gprothetique.hooks import ModelTrackingHooks

@pytest.fixture
def hooks():
    return ModelTrackingHooks()

def test_before_pipeline_run(hooks):
    run_params = {"session_id": "test_session"}
    hooks.before_pipeline_run(run_params)
    active_run = mlflow.active_run()
    assert active_run is not None
    assert active_run.info.run_name == "test_session"
    mlflow.end_run()

def test_after_pipeline_run(hooks):
    mlflow.start_run(run_name="test_session")
    hooks.after_pipeline_run()
    assert mlflow.active_run() is None

# Note: For after_node_run, it requires a more complex setup involving a Node and outputs.
