import sys
from unittest.mock import MagicMock

# Mock comet_ml module before imports
mock_comet = MagicMock()
sys.modules["comet_ml"] = mock_comet

from src.offline.training.comet_tracker import CometTracker  # noqa: E402


def test_comet_tracker_unconfigured():
    # If no api_key, should return None and not raise
    tracker = CometTracker(api_key=None)
    exp = tracker.start_experiment()
    assert exp is None


def test_comet_tracker_configured():
    mock_exp = MagicMock()
    mock_comet.Experiment.return_value = mock_exp

    tracker = CometTracker(api_key="test_comet_api_key", project_name="my_proj")
    exp = tracker.start_experiment()

    assert exp == mock_exp
    mock_comet.Experiment.assert_called_once_with(
        api_key="test_comet_api_key",
        project_name="my_proj",
        workspace=None,
        auto_metric_logging=True,
    )

    tracker.log_metrics({"loss": 0.5}, step=1)
    mock_exp.log_metrics.assert_called_once_with({"loss": 0.5}, step=1)

    tracker.log_parameters({"lr": 2e-4})
    mock_exp.log_parameters.assert_called_once_with({"lr": 2e-4})

    tracker.end_experiment()
    mock_exp.end.assert_called_once()
