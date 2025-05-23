from dags.data_utils.ingestion import fetch_data_from_google_sheets
from dags.data_utils.feature_engineering import create_features
import numpy as np
import pandas as pd

sheet_id = "1yjmPxKbNBRD6DACtkq4l_Xp9O7ldmWujypKE9NhC6Z0"
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


def fill_empty(df):
    df_numerical = df.select_dtypes(include='number')
    df_numerical['time'] = df['time']
    df = df_numerical
    df['match_match_price'] = df['match_match_price'].replace(0, np.nan)
    df = df.ffill()
    df = df.bfill()
    df = df.fillna(0)
    df.dropna(axis=1)
    return df


def combine_dataset(df, feature_df):
    df = fill_empty(df)

    df['time'] = pd.to_datetime(df['time'])

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
    df = pd.merge(df, feature_df, on='time', how='left')
    df = fill_empty(df)
    # Feature set
    return df


def test_data_ingestion():
    df = fetch_data_from_google_sheets(sheet_id, expected_columns, 'Sheet5')
    assert len(df) == 30
    global _df  # temporary global to share with next test
    _df = df


def test_data_engineer():
    feature_df = create_features(_df)
    df = combine_dataset(_df, feature_df)
    expected_features = [
        'listing_ceiling', 'listing_floor', 'listing_ref_price', 'listing_listed_share',
        'match_match_vol', 'match_accumulated_volume', 'match_accumulated_value', 'match_avg_match_price',
        'match_highest', 'match_lowest', 'match_foreign_sell_volume', 'match_foreign_buy_volume',
        'match_current_room', 'match_total_room', 'match_reference_price',
        'mid_price', 'microprice', 'spread_l1', 'relative_spread_l1', 'total_bid_volume_3lv',
        'total_ask_volume_3lv', 'market_depth_value_bid', 'market_depth_value_ask'
    ]
    df = df[expected_features]
    assert len(df) == 30
    assert len(df.columns) == 23
