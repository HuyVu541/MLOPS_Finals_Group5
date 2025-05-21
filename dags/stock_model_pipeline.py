from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
from model_utils.model_engineering import train_model
from model_utils.model_validation import evaluate_model
from model_utils.model_deployment import register_model, compare_models
import os

# ==== Default DAG Config ====
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

USER = os.getenv("USER")

# ==== DAG Definition ====
with DAG(
    dag_id="stock_model_pipeline",
    default_args=default_args,
    description="Train, evaluate, and register LSTM model",
    schedule_interval='@weekly',
    start_date=datetime(2025, 4, 1),
    catchup=False,
    tags=["model_pipeline", "LSTM"],
) as dag:

    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    # mlflow.set_tracking_uri("http://localhost:5000")

    # 1. Train model
    def _train_model(**kwargs):
        result = train_model(
            model_name="LSTM",
            raw_db_name='raw_data',
            raw_table_name='raw_data',
            feature_db_name='feature_db',
            feature_table_name='stock_features'
        )
        kwargs['ti'].xcom_push(key="run_id", value=result["run_id"])
        kwargs['ti'].xcom_push(key="model_uri", value=result["model_uri"])
        print(result["model_uri"])

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        provide_context=True
    )

    # 2. Evaluate model
    def _evaluate_model(**kwargs):
        run_id = kwargs['ti'].xcom_pull(task_ids="train_model", key="run_id")
        result = evaluate_model(
            run_id=run_id,
            raw_db_name='raw_data',
            raw_table_name='raw_data',
            feature_db_name='feature_db',
            feature_table_name='stock_features'
        )
        kwargs['ti'].xcom_push(key="run_id", value=result["run_id"])
        kwargs['ti'].xcom_push(key="model_uri", value=result["model_uri"])

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
        provide_context=True
    )

    # 3. Compare model performance
    def _compare_models(**kwargs):
        run_id = kwargs['ti'].xcom_pull(task_ids="evaluate_model", key="run_id")
        result, best_old_run_id = compare_models(
            new_run_id=run_id,
            experiment_name="LSTM",
            metric_key="mape"
        )
        if result:
            print('New model better.')
            kwargs['ti'].xcom_push(key='best_run_id', value=run_id)
        else:
            print('New model not better. Using old model.')
            kwargs['ti'].xcom_push(key='best_run_id', value=best_old_run_id)

    compare_model_task = PythonOperator(
        task_id="compare_model",
        python_callable=_compare_models,
        provide_context=True
    )

    # 4a. Register model if it's better
    def _register_model(**kwargs):
        best_run_id = kwargs['ti'].xcom_pull(task_ids='compare_model', key='best_run_id')
        run_id = kwargs['ti'].xcom_pull(task_ids='train_model', key="run_id")
        if run_id == best_run_id:
            register_model(
                run_id=run_id,
                model_name="LSTM",
                tags={"version": "auto", "source": "airflow"}
            )

    register_model_task = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
        provide_context=True
    )

    # # 6. Serve model via API (using FastAPI or Flask)
    # def _serve_model(**kwargs):
    #     best_run_id = kwargs['ti'].xcom_pull(task_ids='compare_model', key="best_run_id")
    #     model_serving_setup(run_id=best_run_id)

    # serve_model_task = PythonOperator(
    #     task_id="serve_model",
    #     python_callable=_serve_model,
    #     provide_context=True
    # )

    # # 7. Model rollback if new model fails
    # def _rollback_model(**kwargs):
    #     previous_model_uri = kwargs['ti'].xcom_pull(task_ids="register_model", key="model_uri")
    #     rollback_model(model_uri=previous_model_uri, model_name='LSTM')

    # rollback_model_task = PythonOperator(
    #     task_id="rollback_model",
    #     python_callable=_rollback_model,
    #     provide_context=True
    # )

    # ==== Flow ==== #
    # train_model_task >> evaluate_model_task >> compare_model_task >> register_model_task >> serve_model_task
    train_model_task >> evaluate_model_task >> compare_model_task >> register_model_task
    logging.info('Model pipeline completed.')
