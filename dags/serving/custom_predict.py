import requests
import mlflow
import sys
sys.path.append('/home/huyvu/airflow/dags')
from model_utils.general_utils import construct_dataset
import json

import os

os.environ['DB_USER'] = 'huyvu'
os.environ['DB_PASSWORD'] = 'password'

def custom_predict():
    # The API endpoint


    mlflow.set_tracking_uri("file:/home/huyvu/airflow/mlruns")

    raw_db_name='raw_data'
    raw_table_name='raw_data'
    feature_db_name='feature_db'
    feature_table_name='stock_features'

    df = construct_dataset(raw_db_name, raw_table_name, feature_db_name, feature_table_name, limit = 30)
    df = df.drop(columns = ['time', 'match_match_price'])

    X = df.values.tolist()

    # Send the request
    files = {"file": ("input.json", json.dumps(X), "application/json")}
    res = requests.post("http://127.0.0.1:8000/predict", files=files)
    y_pred = res.json()['predictions']
    # y_pred = res.json()

    print(y_pred)
    return y_pred

if __name__ == '__main__':
    custom_predict()
