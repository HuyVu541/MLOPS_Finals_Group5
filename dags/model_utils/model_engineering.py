import mlflow
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import joblib
import mlflow.keras
import time
import random
import tensorflow as tf
from model_utils.general_utils import construct_dataset
import os
import logging

USER = os.getenv("USER")


def create_sequences(X, y, ts):
    Xs, ys = [], []
    for i in range(ts, len(X) - ts):
        Xs.append(X[i - ts:i])
        ys.append(y[i:i + ts])
    return np.array(Xs), np.array(ys)


def create_lstm_model(input_shape, num_layers=1, units=256, dropout_rate=0.2, time_steps=30):
    seed = int(time.time()) % (2**32 - 1)
    print('Seed: ', seed)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers based on num_layers
    for _ in range(num_layers - 1):  # subtract 1 because the first layer is already added
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))

    # Final LSTM layer (without return_sequences)
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))

    # Dense layer
    model.add(Dense(time_steps))

    # Compile model
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])

    return model


def train_model(
        model_name,
        raw_db_name,
        raw_table_name,
        feature_db_name,
        feature_table_name,
        num_layers=1):
    # mlflow.set_tracking_uri(f"file:/home/{USER}/airflow/mlruns")
    # mlflow.set_tracking_uri("file:/opt/airflow/mlruns")
    mlflow.set_tracking_uri("http://mlflow:5000")

    print(mlflow.get_tracking_uri())
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Testing
    df = construct_dataset(raw_db_name, raw_table_name, feature_db_name, feature_table_name)

    FEATURES = list(df.columns.values)
    FEATURES.remove('time')
    TARGET = ['match_match_price']
    FEATURES.remove('match_match_price')
    print("AWS_SHARED_CREDENTIALS_FILE:", os.getenv("AWS_SHARED_CREDENTIALS_FILE"))
    logging.info('Creating experiment')

    # 2. Tạo experiment và start run
    experiment_name = "LSTM"
    experiment = mlflow.get_experiment_by_name("LSTM")
    if experiment is None:
        mlflow.create_experiment("LSTM", artifact_location=f"s3://mlops-group5/mlruns/{experiment_name}")
    mlflow.set_experiment("LSTM")

    logging.info('Starting run')

    with mlflow.start_run() as run:
        mlflow.set_tag("phase", "training")
        n = len(df)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)

        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]

        feat_scaler = StandardScaler()
        tgt_scaler = StandardScaler()

        X_train_s = train_df[FEATURES].values
        X_val_s = val_df[FEATURES].values
        X_train_s = feat_scaler.fit_transform(X_train_s)
        X_val_s = feat_scaler.transform(X_val_s)

        y_train_s = train_df[TARGET].values.flatten()
        y_val_s = val_df[TARGET].values.flatten()
        y_train_s = tgt_scaler.fit_transform(y_train_s.reshape(-1, 1))

        TIME_STEPS = 30
        print("TIME_STEPS =", TIME_STEPS)

        X_train, y_train = create_sequences(X_train_s, y_train_s, TIME_STEPS)
        X_val, y_val = create_sequences(X_val_s, y_val_s, TIME_STEPS)
        y_train = y_train.reshape(-1, 30)
        y_val = y_val.reshape(-1, 30)

        # 3. Train
        n_features = X_train.shape[2]
        model = create_lstm_model(input_shape=(TIME_STEPS, n_features), num_layers=num_layers)
        model.fit(X_train, y_train, epochs=5, batch_size=256, verbose=2)

        y_pred = tgt_scaler.inverse_transform(model.predict(X_val))
        print("X shape: ", X_val.shape)
        print("y_pred shape: ", y_pred.shape)
        print("y_val shape: ", y_val.shape)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)

        print('MAE: ', mae)
        print('MSE: ', mape)
        print('MAPE: ', mape)

        metrics = {'mae': mae, 'mape': mape, 'mse': mse}
        mlflow.log_metrics(metrics)

        # 4. Log params/metrics tuỳ thích
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("epochs", 5)
        # mlflow.log_param("lr", 2e-5)

        signature = infer_signature(X_val, model.predict(X_val))

        mlflow.keras.log_model(model, artifact_path=model_name, signature=signature)

        # Save locally
        joblib.dump(feat_scaler, "feat_scaler.pkl")
        joblib.dump(tgt_scaler, "tgt_scaler.pkl")

        # Log to MLflow
        mlflow.log_artifact("feat_scaler.pkl", artifact_path="feat_scaler")
        mlflow.log_artifact("tgt_scaler.pkl", artifact_path="tgt_scaler")

        os.remove("feat_scaler.pkl")
        os.remove("tgt_scaler.pkl")

    # 7. Trả về run_id & các URI nếu cần
    run_id = run.info.run_id
    print(run_id)
    # pytorch_uri = f"runs:/{run_id}/{model_name}_pytorch"
    model_uri = f"runs:/{run_id}/{model_name}"
    return {"run_id": run_id, 'model_uri': model_uri}
