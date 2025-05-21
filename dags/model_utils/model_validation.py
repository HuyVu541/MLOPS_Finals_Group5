import pandas as pd
import numpy as np
import mlflow.keras
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import mlflow
from model_utils.general_utils import construct_dataset
import joblib
import os

USER = os.getenv('USER')


def create_sequences(X, y, ts=30):
    Xs, ys = [], []
    for i in range(ts, len(X) - ts):
        Xs.append(X[i - ts:i])
        ys.append(y[i:i + ts])
    return np.array(Xs), np.array(ys)


def evaluate_model(
        run_id: str,
        raw_db_name: str,
        raw_table_name: str,
        feature_db_name: str,
        feature_table_name: str):
    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    # mlflow.set_tracking_uri("http://localhost:5000")
    """
    Validate the model using test data from PostgreSQL.

    Args:
        model_path (str): MLflow model URI (pyfunc), e.g. "runs:/<run_id>/distilbert_sentiment"
        db_uri (str): Database URI for PostgreSQL.
        query (str): SQL query to fetch test data.

    Returns:
        None
    """
    experiment_name = "LSTM"
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(experiment_name)
    # 1. Load test data
    with mlflow.start_run() as run:
        mlflow.set_tag("phase", "validating")
        # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
        # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
        # mlflow.set_tracking_uri("http://localhost:5000")

        df = construct_dataset(
            raw_db_name,
            raw_table_name,
            feature_db_name,
            feature_table_name,
            startfrom=0.8)

        FEATURES = list(df.columns.values)
        FEATURES.remove('time')
        TARGET = ['match_match_price']
        FEATURES.remove('match_match_price')

        X = df[FEATURES].values
        y = df[TARGET]

        # Load feature scaler
        feat_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/feat_scaler/feat_scaler.pkl")
        feat_scaler = joblib.load(feat_local_path)

        # Load target scaler
        tgt_local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"runs:/{run_id}/tgt_scaler/tgt_scaler.pkl")
        tgt_scaler = joblib.load(tgt_local_path)

        X = feat_scaler.transform(X)

        y = y.values.flatten()
        y = tgt_scaler.transform(y.reshape(-1, 1))

        model_name = 'LSTM'
        # 2. Load the pyfunc model
        model = model = mlflow.keras.load_model(f"runs:/{run_id}/{model_name}")

        X, y = create_sequences(X, y)

        y = y.reshape(-1, 30)

        # print(y[0:10])
        # print(y_pred[0:10])

        # 4. Derive hard predictions
        y_pred = tgt_scaler.inverse_transform(model.predict(X))
        y = tgt_scaler.inverse_transform(y)
        print('y_pred shape:', y_pred.shape)
        print('y shape:', y.shape)

        # 5. Compute metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)

        # 6. Compiling metrics
        metrics = {'mae': mae, 'mape': mape, 'mse': mse}

        # 7. Logging to MLFLOW
        mlflow.log_metrics(metrics)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", 5)
        # mlflow.log_param("lr", 2e-5)

        mlflow.keras.log_model(model, artifact_path=model_name)

        # Save locally
        joblib.dump(feat_scaler, "feat_scaler.pkl")
        joblib.dump(tgt_scaler, "tgt_scaler.pkl")

        # Log to MLflow
        mlflow.log_artifact("feat_scaler.pkl", artifact_path="feat_scaler")
        mlflow.log_artifact("tgt_scaler.pkl", artifact_path="tgt_scaler")

        # 8. Print & save report
        print('MAE: ', mae)
        print('MSE: ', mape)
        print('MAPE: ', mape)

        report = {
            "mae": mae,
            "mse": mse,
            "mape": mape,
        }
        pd.DataFrame([report]).to_csv("validation_report.csv", index=False)
        print("Saved validation_report.csv")
    run_id = run.info.run_id
    # pytorch_uri = f"runs:/{run_id}/{model_name}_pytorch"
    model_uri = f"runs:/{run_id}/{model_name}"
    return {"run_id": run_id, 'model_uri': model_uri}
