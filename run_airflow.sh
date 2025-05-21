#!/bin/bash

# Set environment variables for Airflow and PostgreSQL
export DB_USER="huyvu"
export DB_PASSWORD="password"
export AIRFLOW_HOME="$HOME/airflow"  # Set your custom AIRFLOW_HOME if needed

# Initialize Airflow database (only needs to be run once after installing Airflow)
airflow db init

# Start the Airflow scheduler in the background
nohup airflow scheduler &

# Start the Airflow webserver
airflow webserver 
