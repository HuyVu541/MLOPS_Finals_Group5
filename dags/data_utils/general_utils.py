import pandas as pd
import os
from sqlalchemy import create_engine  # Using SQLAlchemy for easier type mapping
import psycopg2

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
USER = os.getenv("USER")
PG_HOST = os.environ.get("PG_HOST", "localhost")


def load_data(query: str, db_uri: str):
    engine = create_engine(db_uri)
    conn = engine.raw_connection()
    df = pd.read_sql(query, con=conn)
    conn.close()
    return df


def _create_table_if_not_exists(db_name, df, table_type):
    db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{PG_HOST}:5432/{db_name}"
    db_config = {
        'dbname': db_name,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': PG_HOST,  # Use 'localhost' or your DB host
        'port': 5432  # Default PostgreSQL port
    }
    conn = psycopg2.connect(**db_config)
    conn.cursor()
    # === 1. Create the table based on DataFrame columns ===

    # Generate column definitions dynamically based on DataFrame types
    columns = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == 'int64':
            col_type = 'BIGINT'
        elif dtype == 'float64':
            col_type = 'DOUBLE PRECISION'
        elif dtype == 'datetime64[ns]':
            col_type = 'TIMESTAMP'
        else:
            col_type = 'TEXT'  # Default to TEXT for string-like columns

        columns.append(f"{col} {col_type}")

    # Combine columns to form the CREATE TABLE statement
    create_table_query = f'''
    CREATE TABLE IF NOT EXISTS {table_type} (
        {', '.join(columns)}
    );
    '''
    engine = create_engine(db_uri)

    with engine.connect() as conn:
        conn.execute(create_table_query)  # Creating the table if it doesn't exist


def insertIntoTable(db_name, df, table):
    db_config = {
        'dbname': db_name,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': PG_HOST,
        'port': 5432
    }

    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # # Fetch existing 'time' values
    # cur.execute(f"SELECT * FROM {table}")
    # existing_times = set(str(row[0]) for row in cur.fetchall())

    db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{PG_HOST}:5432/{db_name}"
    query = f"SELECT * FROM {table}"

    existing_df = load_data(query, db_uri)

    # Filter out rows with existing 'time' values
    new_rows = df[~df['time'].isin(existing_df['time'])]
    print(new_rows)

    if new_rows.empty:
        print("No new rows to insert.")
        return

    tuples = [tuple(x) for x in new_rows.to_numpy()]
    cols = ', '.join(new_rows.columns)
    placeholders = ', '.join(['%s'] * len(new_rows.columns))
    query = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

    try:
        cur.executemany(query, tuples)
        conn.commit()
    except Exception as error:
        print("Error:", error)
        conn.rollback()
    finally:
        cur.close()
        conn.close()

    return len(new_rows)
