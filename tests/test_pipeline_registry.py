from src.gprothetique.pipeline_registry import register_pipelines

def test_register_pipelines():
    pipelines = register_pipelines()
    assert "dp" in pipelines
    assert "ds" in pipelines
    assert "__default__" in pipelines
