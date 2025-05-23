FROM apache/airflow:2.10.5-python3.10

USER root
WORKDIR /opt/airflow

# Install system dependencies
RUN apt-get update && apt-get install -y libpq-dev

# Copy source code
#COPY . /opt/airflow

# Install Python dependencies as airflow user

COPY --chown=airflow:airflow ./dags /opt/airflow/dags
COPY --chown=airflow:airflow requirements.txt /requirements.txt
COPY --chown=airflow:airflow ./mlflow /opt/airflow/mlflow
COPY --chown=airflow:airflow ./postgre /opt/airflow/postgre
COPY --chown=airflow:airflow ./fastapi_app /opt/airflow/fastapi_app

USER airflow

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt


