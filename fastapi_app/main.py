import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import json
import numpy as np
import logging
# Load the model once at startup
import os

MLFLOW_HOST = os.getenv('MLFLOW_HOST', 'mlflow')
mlflow.set_tracking_uri(f"http://{MLFLOW_HOST}:5000")

client = MlflowClient()

# Replace with your experiment name or ID
experiment_name = "LSTM"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Search for best run by accuracy
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.phase = 'validating' and attributes.status = 'FINISHED'",
    order_by=["metrics.mape DESC"],
    max_results=1,
)


best_run = runs[0]
run_id = best_run.info.run_id
logging.info(f"{run_id}")

feat_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/feat_scaler/feat_scaler.pkl")
feat_scaler = joblib.load(feat_local_path)

tgt_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/tgt_scaler/tgt_scaler.pkl")
tgt_scaler = joblib.load(tgt_local_path)

model_name = 'LSTM'
# 2. Load the pyfunc model
model = mlflow.keras.load_model(f"runs:/{run_id}/{model_name}")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Model Serving API. Use POST /predict with a JSON file to get predictions."}

# Prediction endpoint: accepts a JSON file upload


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded JSON file
    contents = await file.read()
    data = np.array(json.loads(contents.decode("utf-8")))

    if data.shape != (30, 23):
        raise HTTPException(status_code=400, detail=f"Expected input shape (30, 23), got {data.shape}")

    # Convert the input data to a numpy array and reshape for the model
    data = feat_scaler.inverse_transform(data)
    data = data.reshape(1, 30, 23)  # Reshape based on your model's expected input

    # Make predictions using the model
    predictions = model.predict(data)
    predictions = tgt_scaler.inverse_transform(predictions)

    # Return predictions in JSON format
    return {"predictions": predictions.tolist()}

# To launch app: uvicorn app:app --reload
