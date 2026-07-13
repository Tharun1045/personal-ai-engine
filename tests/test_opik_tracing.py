import sys
from unittest.mock import MagicMock

# Mock opik module before testing
mock_opik_mod = MagicMock()
sys.modules["opik"] = mock_opik_mod

from unittest.mock import patch  # noqa: E402
from src.online.tracing.opik_client import OpikTracer, track  # noqa: E402


@patch("src.online.tracing.opik_client.settings")
def test_opik_tracer_configured(mock_settings):
    mock_settings.OPIK_API_KEY = "test_key"
    mock_settings.OPIK_PROJECT_NAME = "test_proj"

    tracer = OpikTracer()
    assert tracer.client is not None
    mock_opik_mod.Opik.assert_called_once_with(project_name="test_proj")


@patch("src.online.tracing.opik_client.settings")
def test_opik_tracer_unconfigured(mock_settings):
    mock_settings.OPIK_API_KEY = None
    tracer = OpikTracer()
    assert tracer.client is None


@patch("src.online.tracing.opik_client.settings")
def test_opik_track_decorator(mock_settings):
    mock_settings.OPIK_API_KEY = "test_key"

    # Setup mock decorator behavior
    mock_decorator = MagicMock()
    mock_opik_mod.track.return_value = mock_decorator

    @track()
    def my_dummy_func(x):
        return x + 1

    mock_opik_mod.track.assert_called_once()
    mock_decorator.assert_called_once()
