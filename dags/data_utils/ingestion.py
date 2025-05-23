import pandas as pd
import logging
import os
import gspread
from data_utils.general_utils import _create_table_if_not_exists, insertIntoTable


# --- Configuration ---
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
SHEET_NUMBER = os.environ.get("GOOGLE_SHEET_NUMBER", "Sheet4")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_data_from_google_sheets(sheet_id, expected_columns, SHEET_NUMBER = SHEET_NUMBER):
    """
    Fetches data from the configured Google Sheet and converts specific columns to appropriate types.
    """
    if not sheet_id:
        logging.error("Missing required environment variable: GOOGLE_SHEET_ID")
        raise ValueError("GOOGLE_SHEET_ID environment variable not set.")

    try:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={SHEET_NUMBER}"
        df = pd.read_csv(url)
    except gspread.exceptions.APIError as e:
        logging.error(f"Google Sheets API error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error connecting to or reading from Google Sheets: {e}")
        raise

    if len(df) == 0:
        logging.warning("No data found in the Google Sheet.")
        return pd.DataFrame()

    # df = pd.DataFrame(data)
    logging.info(f"Fetched {len(df)} rows from Google Sheet.")

    # Verify required columns exist
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing expected columns in fetched data: {missing_cols}.")
        raise

    # Convert empty strings to None
    df = df.replace('', None)

    # --- Specific Data Type Conversion ---
    logging.info("Converting data types for specific columns...")

    # logging.info(f"Data types converted. DataFrame info:\n")
    # df.info(verbose=True, show_counts=True)
    logging.info(f"DataFrame head:\n{df.head().to_string()}")

    # Reorder columns to expected order if necessary
    present_expected_cols = [col for col in expected_columns if col in df.columns]
    df = df[present_expected_cols]
    df.rename(columns={'Time': 'time'}, inplace=True)

    return df


def validate_data(df):
    if len(df) == 0:
        logging.warning("Dataframe is empty.")
        return False
        # Depending on requirements, you might return True or False here.
        # Returning False as usually an empty table after ingestion is an issue.

    if df['match_match_price'].isna().any():
        logging.error('Empty values in target column.')
        return False

    return True

# --- Main Function for DAG Task ---


def ingest_data(db_name, sheet_id, table_name):
    """
    Orchestrator function called by the Airflow task.
    Fetches data from Google Sheets, converts types, and appends (deduplicated)
    into the golden SQLite database.
    """
    logging.info("Starting ingestion process (append, deduplicated)...")
    # Define the list of expected columns based on user input
    expected_columns = ['Time', 'listing_symbol', 'listing_ceiling', 'listing_floor',
                        'listing_ref_price', 'listing_stock_type', 'listing_exchange',
                        'listing_trading_status', 'listing_security_status',
                        'listing_last_trading_date', 'listing_listed_share',
                        'listing_sending_time', 'listing_type', 'listing_organ_name',
                        'listing_mapping_symbol', 'listing_product_grp_id', 'listing_partition',
                        'listing_index_type', 'listing_trading_date',
                        'listing_lst_trading_status', 'bid_ask_transaction_time',
                        'match_accumulated_value', 'match_accumulated_volume',
                        'match_accumulated_value_g1', 'match_accumulated_volume_g1',
                        'match_avg_match_price', 'match_current_room',
                        'match_foreign_buy_volume', 'match_foreign_sell_volume',
                        'match_foreign_buy_value', 'match_foreign_sell_value', 'match_highest',
                        'match_lowest', 'match_match_price', 'match_match_type',
                        'match_match_vol', 'match_sending_time', 'match_total_room',
                        'match_total_buy_orders', 'match_total_sell_orders', 'match_bid_count',
                        'match_ask_count', 'match_underlying', 'match_open_interest',
                        'match_stock_type', 'match_partition', 'match_is_match_price',
                        'match_ceiling_price', 'match_floor_price', 'match_reference_price',
                        'bid_ask_bid_1_price', 'bid_ask_bid_1_volume', 'bid_ask_bid_2_price',
                        'bid_ask_bid_2_volume', 'bid_ask_bid_3_price', 'bid_ask_bid_3_volume',
                        'bid_ask_ask_1_price', 'bid_ask_ask_1_volume', 'bid_ask_ask_2_price',
                        'bid_ask_ask_2_volume', 'bid_ask_ask_3_price', 'bid_ask_ask_3_volume']
    try:
        print('Fetching Data')
        df = fetch_data_from_google_sheets(sheet_id, expected_columns)
        if not validate_data(df):
            logging.error('Validation failed.')
            raise
        print('Creating table')
        _create_table_if_not_exists(db_name, df, table_type=table_name)
        print('Inserting Data')
        inserted_count = insertIntoTable(db_name, df, table_name)
        logging.info(
            f"Ingestion process completed. Approximately {inserted_count} new rows inserted.")
    except Exception as e:
        logging.error(f"Ingestion process failed: {e}")
        raise
