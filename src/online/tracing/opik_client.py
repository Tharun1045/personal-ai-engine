import functools
from typing import Any, Callable
from loguru import logger
from src.shared.config import settings


class OpikTracer:
    def __init__(self):
        self.api_key = settings.OPIK_API_KEY
        self.project_name = settings.OPIK_PROJECT_NAME
        self.client = None

        if self.api_key:
            try:
                import opik  # type: ignore

                self.client = opik.Opik(project_name=self.project_name)
                logger.info("Opik tracing initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize Opik client: {e}")


def track() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace functions with Opik if configured, otherwise acts as no-op."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if settings.OPIK_API_KEY:
            try:
                import opik  # type: ignore

                return opik.track()(func)
            except Exception as e:
                logger.warning(f"Failed to apply Opik track decorator: {e}")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator
