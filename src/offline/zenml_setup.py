from zenml import pipeline, step
from zenml.client import Client


@step
def minimal_smoke_step() -> str:
    return "Smoke step successful"


@pipeline
def minimal_smoke_pipeline():
    minimal_smoke_step()


def verify_zenml_server() -> bool:
    try:
        client = Client()
        # Ensure we can connect to the ZenML server URL if it's set correctly
        store_info = client.zen_store.get_store_info()
        return store_info is not None
    except Exception:
        return False
