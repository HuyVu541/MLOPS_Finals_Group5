import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Replace with your experiment name or ID
experiment_name = "LSTM"
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Search for best run by accuracy
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string="tags.phase = 'validating' and attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)


best_run = runs[0]
best_model_uri = f"runs:/{best_run.info.run_id}/model"

# Load the best model
model = mlflow.pyfunc.load_model(best_model_uri)
