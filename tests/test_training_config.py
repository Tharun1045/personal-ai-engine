import sys
from src.offline.training.training_config import TrainingConfig
from src.offline.training.trainer import train_model


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.max_seq_length == 2048
    assert config.load_in_4bit is True
    assert config.lora_r == 16
    assert config.learning_rate == 2e-4


def test_trainer_windows_skip():
    # If we are on Windows, train_model should gracefully return False (skip)
    config = TrainingConfig()
    if sys.platform == "win32":
        result = train_model(config, "mock_dataset")
        assert result is False
    else:
        # On Linux/WSL, fast language model isn't installed in mock, so we expect import error or similar
        pass
