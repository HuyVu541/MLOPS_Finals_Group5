from fastapi.testclient import TestClient
from dags.serving.app import app  # your FastAPI instance

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
