# --- START OF FILE data_pipeline.py ---

from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.models import Param # Import Param for better parameter definition
import os
from data_utils.ingestion import ingest_data
from data_utils.feature_engineering import engineering_features

# --- Configuration ---

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
PG_GOLDEN_DBNAME = os.environ.get("PG_GOLDEN_DBNAME", "raw_data")
PG_FEATURE_DBNAME = os.environ.get("PG_FEATURE_DBNAME", "feature_db")
os.environ['GOOGLE_SHEET_ID'] = "1yjmPxKbNBRD6DACtkq4l_Xp9O7ldmWujypKE9NhC6Z0"

# Helper to build URI (sensitive info like password better handled by Airflow Connections)
def build_pg_uri(host, port, user, password, dbname):
    if password:
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    else: # Handle case where password might be empty or managed differently
        return f"postgresql://{user}@{host}:{port}/{dbname}"

default_golden_uri = build_pg_uri(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_GOLDEN_DBNAME)
default_feature_uri = build_pg_uri(PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_FEATURE_DBNAME)


# Define the parameters for the DAG
params = {
    "golden_database_uri": Param(default_golden_uri, type="string", title="Golden DB URI", description="SQLAlchemy URI for the Golden PostgreSQL database."),
    "feature_store_uri": Param(default_feature_uri, type="string", title="Feature DB URI", description="SQLAlchemy URI for the Feature Store PostgreSQL database."),
    "golden_table_name": Param("raw_data", type="string", title="Golden Table Name"),
    "feature_store_table_name": Param("stock_features", type="string", title="Feature Table Name")
    # "unique_key_column": Param("Time", type="string", title="Unique Key Column(s)", description="Comma-separated if multiple keys."), # Allow comma separated for multiple keys
}

# Set the default parameters for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
with DAG(
    "stock_data_pipeline", 
    default_args=default_args,
    params=params,
    tags=["data_pipeline", "google_sheets", "postgres", "append"], 
    description="Hourly pipeline: Google Sheets -> Append PostgreSQL Golden (Deduplicated) -> PostgreSQL Features", 
    schedule_interval='@daily', # Use cron preset
    start_date=datetime(2025, 4, 25),
    catchup=False,
) as dag:

    @task
    def task_ingest_data(**context):
        """Fetches data from Google Sheets and appends it (deduplicated) to the Golden PostgreSQL DB."""
        conf = context.get("dag_run").conf if context.get("dag_run") else context["params"]
        conf.get("golden_database_uri") 
        golden_table_name = conf.get("golden_table_name")
        
        sheet_id = os.environ['GOOGLE_SHEET_ID'] 

        ingest_data(golden_table_name, sheet_id, 'raw_data')
        print(f"Ingestion task finished. Data appended to Golden DB (URI used), table {golden_table_name}")

    # @task
    # def task_validate_table(**context):
    #     """Validates the data in the Golden PostgreSQL DB."""
    #     conf = context.get("dag_run").conf if context.get("dag_run") else context["params"]
    #     golden_table_name = conf.get("golden_table_name")

    #     is_valid = validate_table(PG_GOLDEN_DBNAME, golden_table_name) 
    #     if not is_valid:
    #         raise ValueError(f"Validation failed for table '{golden_table_name}' in Golden DB (URI used)")
    #     print(f"Validation task finished successfully for Golden DB (URI used), table {golden_table_name}")

    @task
    def task_engineer_features(**context):
        """Reads data from Golden PostgreSQL DB, engineers features, and stores them in Feature Store PostgreSQL DB."""
        conf = context.get("dag_run").conf if context.get("dag_run") else context["params"]
        conf.get("golden_database_uri") # Use URI param
        golden_table_name = conf.get("golden_table_name")
        conf.get("feature_store_uri") # Use URI param
        feature_store_table_name = conf.get("feature_store_table_name")


        # Feature engineering overwrites the feature table each time based on the current golden data
        engineering_features(
            golden_database=PG_GOLDEN_DBNAME, # Pass URI
            golden_table_name=golden_table_name,
            feature_store_database=PG_FEATURE_DBNAME, # Pass URI
            feature_store_table_name=feature_store_table_name,
        )
        print(f"Feature engineering task finished. Features stored in Feature Store DB (URI used), table {feature_store_table_name}")

    # Define the task dependencies
    ingest_task = task_ingest_data()
    # validate_task = task_validate_table()
    engineer_features_task = task_engineer_features()


    # ingest_task >> validate_task >> engineer_features_task
    ingest_task >> engineer_features_task
