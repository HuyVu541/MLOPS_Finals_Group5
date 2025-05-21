
from fastapi import FastAPI, File, UploadFile
import mlflow
import json
import numpy as np
# from serving.custom_predict import predict
import joblib
# Load the model once at startup
mlflow.set_tracking_uri("file:/home/huyvu/airflow/mlruns")

run_id = 'e8ac2630d1804c42b16198a592adc6ac'
model_name = 'LSTM'

model = mlflow.keras.load_model(f"runs:/{run_id}/{model_name}")

# Load feature scaler

feat_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/feat_scaler/feat_scaler.pkl")
feat_scaler = joblib.load(feat_local_path)

# Load target scaler
tgt_local_path = mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/tgt_scaler/tgt_scaler.pkl")
tgt_scaler = joblib.load(tgt_local_path)


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

    if data.shape != (30, 26):
        raise ValueError(f"Expected input shape (30, 26), got {data.shape}")

    # Convert the input data to a numpy array and reshape for the model
    data = feat_scaler.inverse_transform(data)
    data = data.reshape(1, 30, 26)  # Reshape based on your model's expected input

    # Make predictions using the model
    predictions = model.predict(data)
    predictions = tgt_scaler.inverse_transform(model.predict(data))

    # Return predictions in JSON format
    return {"predictions": predictions.tolist()}

# To launch app: uvicorn app:app --reload
