# Serve a trained model using MLflow

import mlflow
import mlflow.keras
import os


USER = os.getenv('USER')


def model_serving_setup(run_id):
    """Set up model serving infrastructure."""
    print("Setting up model serving...")

    # In a real scenario, this might:
    # 1. Deploy to a Flask/FastAPI service
    # 2. Update a Kubernetes deployment
    # 3. Push to a model serving platform like SageMaker

    # For this example, we'll simulate deployment by downloading the model
    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_tracking_uri("http://local_host:5000")

    # Simulate preparing a serving environment
    serving_dir = os.path.expanduser("~/airflow/dags/serving")
    os.makedirs(serving_dir, exist_ok=True)

    # Full path for app.py
    serving_path = os.path.join(serving_dir, "app.py")

    # Download the model to the serving directory
    # In a real scenario, you might use MLflow's built-in serving capabilities
    # Create a simple prediction script
    prediction_script = f"""
from fastapi import FastAPI, File, UploadFile
import mlflow
import pandas as pd
import json
import numpy as np
import sys
import boto3

# from serving.custom_predict import predict
import joblib
# Load the model once at startup
# mlflow.set_tracking_uri("file:/home/{USER}/airflow/mlruns")
# mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
# mlflow.set_tracking_uri("http://localhost:5000")

run_id = '{run_id}'
model_name = 'LSTM'

model = mlflow.keras.load_model(f"runs:/{{run_id}}/{{model_name}}")

# Load feature scaler

feat_local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{{run_id}}/feat_scaler/feat_scaler.pkl")
feat_scaler = joblib.load(feat_local_path)

# Load target scaler
tgt_local_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{{run_id}}/tgt_scaler/tgt_scaler.pkl")
tgt_scaler = joblib.load(tgt_local_path)


app = FastAPI()

@app.get("/")
def home():
    return {{"message": "Model Serving API. Use POST /predict with a JSON file to get predictions."}}

# Prediction endpoint: accepts a JSON file upload
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded JSON file
    contents = await file.read()
    data = np.array(json.loads(contents.decode("utf-8")))

    if data.shape != (30,23):
        raise ValueError(f"Expected input shape (30, 23), got {{data.shape}}")

    # Convert the input data to a numpy array and reshape for the model
    data = feat_scaler.inverse_transform(data)
    data = data.reshape(1, 30, 23)  # Reshape based on your model's expected input

    # Make predictions using the model
    predictions = model.predict(data)
    predictions = tgt_scaler.inverse_transform(model.predict(data))

    # Return predictions in JSON format
    return {{"predictions": predictions.tolist()}}

# To launch app: uvicorn app:app --reload
"""

    with open(serving_path, "w") as f:
        f.write(prediction_script)

    print(f"Model serving setup complete at {serving_dir}")

    # return {"serving_dir": serving_dir, "model_uri": model_uri}
