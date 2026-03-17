from semantic_cloud.config import DataConfig, ModelConfig, TrainConfig


def test_default_configs_are_instantiable():
    data = DataConfig()
    model = ModelConfig(model_type="transformer")
    train = TrainConfig()

    assert data.max_length == 40
    assert model.model_type == "transformer"
    assert train.batch_size > 0
