from mlflow.tracking import MlflowClient
import mlflow
import os
import boto3
import logging


USER = os.getenv('USER')

def register_model(run_id: str, model_name: str, tags: dict):
    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_tracking_uri("http://mlflow:5000")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{model_name}"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    # Thêm tag cho phiên bản model đã được đăng ký
    for key, value in tags.items():
        client.set_model_version_tag(name=model_name, version=result.version, key=key, value=value)

    return result.version



def compare_models(new_run_id: str, experiment_name: str, metric_key: str = "mape") -> bool:

    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_tracking_uri("http://mlflow:5000")
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = mlflow.tracking.MlflowClient()

    # Lấy thông tin model mới
    new_metrics = client.get_run(new_run_id).data.metrics
    new_metric_value = new_metrics.get(metric_key, None)
    if new_metric_value is None:
        raise ValueError(f"Metric '{metric_key}' not in run {new_run_id}")

    # Lấy top model trước đó trong cùng experiment
    experiment = client.get_experiment_by_name(experiment_name)
    all_runs = client.search_runs(  
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.phase = 'validating'",
        order_by=[f"metrics.{metric_key} ASC"],
        max_results=5,
    )

    all_runs = [run for run in all_runs if metric_key in run.data.metrics and run.info.run_id != new_run_id]
    logging.info([run.info.run_id for run in all_runs])

    if not all_runs:
        print("No old models found. Using new model.")
        return True, new_run_id

    best_old_run = all_runs[0]
    best_old_value = best_old_run.data.metrics[metric_key]
    best_old_model_uri = f"runs:/{best_old_run.info.run_id}/LSTM"
    print(f"[Old model] {metric_key}: {best_old_value:.4f}")
    print(f"[New model] {metric_key}: {new_metric_value:.4f}")
    
    return new_metric_value < best_old_value, best_old_run.info.run_id