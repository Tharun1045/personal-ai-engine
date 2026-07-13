from typing import Any, Dict, Optional
from loguru import logger
from src.shared.config import settings


class CometTracker:
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: Optional[str] = None,
        workspace: Optional[str] = None,
    ):
        self.api_key = api_key or settings.COMET_API_KEY
        self.project_name = project_name or settings.COMET_PROJECT_NAME
        self.workspace = workspace or settings.COMET_WORKSPACE
        self.experiment = None

    def start_experiment(self) -> Any:
        """Start tracking training run with Comet ML.

        If api_key is not configured, logs a warning and returns None.
        """
        if not self.api_key:
            logger.warning(
                "Comet API key (COMET_API_KEY) is not configured. "
                "Skipping real Comet experiment tracking. (Using mock mode)"
            )
            return None

        try:
            import comet_ml  # type: ignore

            logger.info("Initializing Comet ML experiment...")
            self.experiment = comet_ml.Experiment(
                api_key=self.api_key,
                project_name=self.project_name,
                workspace=self.workspace,
                auto_metric_logging=True,
            )
            return self.experiment
        except Exception as e:
            logger.error(f"Failed to start Comet ML experiment: {e}")
            raise e

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.experiment:
            self.experiment.log_metrics(metrics, step=step)

    def log_parameters(self, params: Dict[str, Any]) -> None:
        if self.experiment:
            self.experiment.log_parameters(params)

    def end_experiment(self) -> None:
        if self.experiment:
            self.experiment.end()
