import pandas as pd
import numpy as np
from sqlalchemy import create_engine # Using SQLAlchemy for easier type mapping
import os 

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
PG_HOST = os.getenv("PG_HOST", 'localhost')
def fill_empty(df):
    df_numerical = df.select_dtypes(include = 'number')
    df_numerical['time'] = df['time']
    df = df_numerical
    df['match_match_price'] = df['match_match_price'].replace(0, np.nan)
    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)
    df.dropna(axis = 1)
    return df  

def construct_dataset(raw_db_name, raw_table_name, feature_db_name, feature_table_name, startfrom = 0, limit = None):
    raw_db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{PG_HOST}:5432/{raw_db_name}"
    feature_db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{PG_HOST}:5432/{feature_db_name}"
    raw_query = f"SELECT * FROM {raw_table_name}"
    feature_query = f"SELECT * FROM {feature_table_name}"

    df = load_data(raw_query, raw_db_uri)
    df = df[int(len(df) * startfrom):]
    df = fill_empty(df)

    df['time'] = pd.to_datetime(df['time'])
    df = fill_empty(df)
    
    FEATURES = [
        'time',
        'listing_ceiling',
        'listing_floor',
        'listing_ref_price',
        'listing_listed_share',

        'match_match_vol',
        'match_accumulated_volume',
        'match_accumulated_value',
        'match_avg_match_price',
        'match_highest',
        'match_lowest',

        'match_foreign_sell_volume',
        'match_foreign_buy_volume',     
        'match_current_room',
        'match_total_room',

        'match_reference_price',
        'match_match_price' 
    ]
    df = df[FEATURES]

    feature_df = load_data(feature_query, feature_db_uri)

    df['time'] = pd.to_datetime(df['time'])
    df = pd.merge(df, feature_df, on='time', how='left')
    df = fill_empty(df)
    # Feature set
    

    if limit:
        return df.tail(limit)
    
    return df

def load_data(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df