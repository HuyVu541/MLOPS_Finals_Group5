import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import psycopg2
from .general_utils import _create_table_if_not_exists, insertIntoTable

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
PG_HOST = os.environ.get("PG_HOST", "localhost")
USER = os.getenv("USER")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_features(df):
    """
    Engineers features based on listing info and order book depth.
    Assumes input DataFrame has relevant columns and 'Time' column is datetime.
    """
    if df.empty:
        logging.warning("Input DataFrame for feature creation is empty. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.copy()
    logging.info("Starting feature engineering...")
    start_time = datetime.now()
    logging.info(f"Before feature engineering: {len(df)} rows")
    # Ensure 'Time' column is datetime type - crucial for time-based features.
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        logging.warning("Converting 'time' column to datetime.")
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isnull().any():
            logging.warning("Some 'time' values could not be parsed and are set to NaT.")

    # Define required columns for safety
    required_cols = [
        'time', 'bid_ask_bid_1_price', 'bid_ask_ask_1_price',
        'bid_ask_bid_1_volume', 'bid_ask_ask_1_volume',
        'bid_ask_bid_2_price', 'bid_ask_bid_2_volume',
        'bid_ask_ask_2_price', 'bid_ask_ask_2_volume',
        'bid_ask_bid_3_price', 'bid_ask_bid_3_volume',
        'bid_ask_ask_3_price', 'bid_ask_ask_3_volume'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns for feature engineering: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    epsilon = 1e-9  # Small value to avoid division by zero

    # --- Feature Engineering Logic ---
    logging.info("Calculating basic price & spread features...")
    df['mid_price'] = (df['bid_ask_bid_1_price'] + df['bid_ask_ask_1_price']) / 2
    vol_sum_l1 = df['bid_ask_bid_1_volume'] + df['bid_ask_ask_1_volume']
    numerator = (df['bid_ask_bid_1_price'] * df['bid_ask_ask_1_volume'] +
                 df['bid_ask_ask_1_price'] * df['bid_ask_bid_1_volume'])
    # Use mid_price as fallback where L1 volume is zero
    df['microprice'] = np.where(vol_sum_l1 > epsilon, numerator / vol_sum_l1, df['mid_price'])

    df['spread_l1'] = df['bid_ask_ask_1_price'] - df['bid_ask_bid_1_price']
    df['relative_spread_l1'] = np.where(
        df['mid_price'] > epsilon,
        df['spread_l1'] /
        df['mid_price'],
        np.nan)

    logging.info("Calculating volume & liquidity features...")
    df['total_bid_volume_3lv'] = df['bid_ask_bid_1_volume'] + \
        df['bid_ask_bid_2_volume'] + df['bid_ask_bid_3_volume']
    df['total_ask_volume_3lv'] = df['bid_ask_ask_1_volume'] + \
        df['bid_ask_ask_2_volume'] + df['bid_ask_ask_3_volume']

    df['market_depth_value_bid'] = (df['bid_ask_bid_1_price'] * df['bid_ask_bid_1_volume'] +
                                    df['bid_ask_bid_2_price'] * df['bid_ask_bid_2_volume'] +
                                    df['bid_ask_bid_3_price'] * df['bid_ask_bid_3_volume'])
    df['market_depth_value_ask'] = (df['bid_ask_ask_1_price'] * df['bid_ask_ask_1_volume'] +
                                    df['bid_ask_ask_2_price'] * df['bid_ask_ask_2_volume'] +
                                    df['bid_ask_ask_3_price'] * df['bid_ask_ask_3_volume'])

    logging.info("Calculating time-based features...")
    if 'time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time_hour'] = df['time'].dt.hour
        df['time_minute'] = df['time'].dt.minute
    else:
        logging.warning("'time' column not available or not datetime type for time-based features.")
        df['time_hour'] = np.nan
        df['time_minute'] = np.nan

    end_time = datetime.now()
    logging.info(f"Feature engineering complete. Duration: {end_time - start_time}")

    # Select only newly created feature columns + timestamp
    feature_columns = [
        'time',  # Keep timestamp for joining/analysis
        'mid_price', 'microprice', 'spread_l1', 'relative_spread_l1',
        'total_bid_volume_3lv', 'total_ask_volume_3lv',
        'market_depth_value_bid', 'market_depth_value_ask',
        'time_hour', 'time_minute'
    ]

    final_features = df[feature_columns].copy()

    # Handle potential infinities resulting from calculations
    final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    logging.info(f"After feature engineering: {len(final_features)} rows.")

    return final_features


# --- Main Function for DAG Task ---
def engineering_features(
        golden_database,
        golden_table_name,
        feature_store_database,
        feature_store_table_name):
    """
    Main feature engineering task called by Airflow.
    Reads from golden DB, creates features, saves to feature store DB.
    """
    logging.info("Starting feature engineering task...")
    logging.info(f"Reading data from Golden DB: {golden_database}, Table: {golden_table_name}")

    db_config = {
        'dbname': golden_database,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': PG_HOST,
        'port': 5432
    }

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # Fetch existing 'time' values
    cur.execute(f"SELECT * FROM {golden_table_name}")
    df = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

    features_df = create_features(df)

    conn.close()

    print('Creating table')
    _create_table_if_not_exists(feature_store_database, features_df, feature_store_table_name)
    print('Inserting Data')
    inserted_count = insertIntoTable(feature_store_database, features_df, feature_store_table_name)
    logging.info(f"Ingestion process completed. Approximately {inserted_count} new rows inserted.")


# --- Command Line Execution (for testing) ---
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Feature Engineering Task")
#     parser.add_argument('--input-db', required=True, help="Path to the input SQLite database (Golden DB).")
#     parser.add_argument('--input-table', required=True, help="Table name in the input database.")
#     parser.add_argument('--output-db', required=True, help="Path to the output SQLite database (Feature Store DB).")
#     parser.add_argument('--output-table', required=True, help="Table name for features in the output database.")

#     args = parser.parse_args()

#     # Create directory for output DB if it doesn't exist
#     os.makedirs(os.path.dirname(args.output_db), exist_ok=True)

#     # Run the feature engineering task
#     engineering_features(
#         golden_database=args.input_db,
#         golden_table_name=args.input_table,
#         feature_store_database=args.output_db,
#         feature_store_table_name=args.output_table
#     )
