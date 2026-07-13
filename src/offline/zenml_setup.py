from zenml import pipeline, step
import urllib.request
from src.shared.config import settings


@step
def minimal_smoke_step() -> str:
    return "Smoke step successful"


@pipeline
def minimal_smoke_pipeline():
    minimal_smoke_step()


def verify_zenml_server() -> bool:
    try:
        url = settings.ZENML_STORE_URL
        if url.startswith("sqlite"):
            # Local fallback or misconfigured URL in tests
            return True
        req = urllib.request.Request(f"{url}/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False
