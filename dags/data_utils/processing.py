import pandas as pd
import logging
import os
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
USER = os.getenv("USER")
PG_HOST = os.getenv("PG_HOST", 'localhost')

def validate_table(database_file, table_name):
    """
    Validate if the table in the SQLite database exists and contains data.
    Prints a sample and returns True if valid, False otherwise.

    Args:
        database_file (str): Path to the SQLite database file.
        table_name (str): Name of the table to validate.
    Returns:
        bool: True if table exists and has data, False otherwise.
    """
    logging.info(f"Starting validation for table '{table_name}' in database '{database_file}'...")

    # if not os.path.exists(database_file):
    #     logging.error(f"Database file '{database_file}' does not exist.")
    #     return False

    conn = None
    try:
        db_config = {
        'dbname': database_file,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'host': PG_HOST,
        'port': 5432
    }

        conn = psycopg2.connect(**db_config)

        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
        
        logging.info(f"Number of records found in '{table_name}': {len(df)}")

        if len(df) == 0:
            logging.warning(f"Table '{table_name}' exists but is empty.")
            # Depending on requirements, you might return True or False here.
            # Returning False as usually an empty table after ingestion is an issue.
            return False
        
        if df['match_match_price'].isna().any():
            logging.info('Empty price.')
            return False

        # Preview data
        sample_query = f"SELECT * FROM {table_name} LIMIT 5" # Table name cannot be parameterized here safely
        sample_data = pd.read_sql_query(sample_query, conn)
        print(f"\nSample data from '{table_name}':\n{sample_data.to_string()}") # Use to_string for better console output

        logging.info(f"Validation successful for table '{table_name}'.")
        return True

    except Exception as e:
        logging.error(f"Error during validation: {e}")
        return False
    finally:
        if conn:
            conn.close()
            logging.info("Postgresql connection closed.")


# # --- Command Line Execution (for testing) ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Validate and preview data from a table in an SQLite database."
#     )
#     parser.add_argument(
#         "--database", required=True, help="Path to the SQLite database file."
#     )
#     parser.add_argument(
#         "--table", required=True, help="Table name in SQLite database."
#     )

#     args = parser.parse_args()

#     is_valid = validate_table(args.database, args.table)

#     if is_valid:
#         print(f"\nValidation PASSED for table '{args.table}' in '{args.database}'.")
#         exit(0)
#     else:
#         print(f"\nValidation FAILED for table '{args.table}' in '{args.database}'.")
#         exit(1)