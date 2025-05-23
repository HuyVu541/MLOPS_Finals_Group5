from fastapi.testclient import TestClient
from fastapi_app.main import app  # your FastAPI instance
import logging

client = TestClient(app)


def test_home():
    logging.info("Pinging FastAPI server.")
    response = client.get("/")
    assert response.status_code == 200
