from dataclasses import dataclass


@dataclass
class DataConfig:
    min_length: int = 20
    max_length: int = 40
    vocab_size: int = 8000
    train_size: int = 12000
    valid_size: int = 2000
    test_size: int = 2000


@dataclass
class ModelConfig:
    model_type: str
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    num_classes: int = 8
    num_clouds: int = 6


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 8
    seed: int = 7
