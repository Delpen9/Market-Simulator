import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data

def compute_portvals(
    orders_df,
    start_val = 1000000,
    commission = 9.95,
    impact = 0.005,
):
    """
    Computes the portfolio values.

    :param orders_dataframe: Dataframe containing orders information
    :type orders_dataframe: dataframe
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    begin_date_list = [int(date_val) for date_val in str(orders_df.values[0, 0]).split('-')]
    begin_date = dt.datetime(begin_date_list[0], begin_date_list[1], begin_date_list[2])

    end_date_list = [int(date_val) for date_val in str(orders_df.values[-1, 0]).split('-')]
    end_date = dt.datetime(end_date_list[0], end_date_list[1], end_date_list[2])

    unique_symbols = list(orders_df['Symbol'].unique())

    data_1 = get_data(unique_symbols, pd.date_range(begin_date - dt.timedelta(days = 1), end_date))[unique_symbols]
    data_1['Cash'] = 1.0

    data_2 = get_data(unique_symbols, pd.date_range(begin_date - dt.timedelta(days = 1), end_date))[unique_symbols]
    data_2['Cash'] = 0.0
    data_2[unique_symbols] = 0.0
    
    data_3 = get_data(unique_symbols, pd.date_range(begin_date - dt.timedelta(days = 1), end_date))[unique_symbols]
    data_3['Cash'] = 0.0
    data_3.iloc[0, -1] = start_val
    data_3[unique_symbols]= 0.0

    def calculate_data_2(df, df_2, date, symbol, shares, impact, commission, buy_or_sell):
        df.loc[date, symbol] = df.loc[date, symbol] + (-buy_or_sell) * shares
        df.loc[date, 'Cash'] = df.loc[date, 'Cash'] + (df_2.loc[date, symbol] * shares * (buy_or_sell - impact)) - commission
    
    sell_orders_df = orders_df.where(orders_df['Order'] == 'SELL').dropna()[['Date', 'Symbol', 'Shares']]
    sell_orders_df.apply(lambda sell_order: calculate_data_2(
            data_2,
            data_1,
            sell_order['Date'],
            sell_order['Symbol'],
            sell_order['Shares'],
            impact,
            commission,
            1
        ),
        axis = 1
    )

    buy_orders_df = orders_df.where(orders_df['Order'] != 'SELL').dropna()[['Date', 'Symbol', 'Shares']]
    buy_orders_df.apply(lambda buy_order: calculate_data_2(
            data_2,
            data_1,
            buy_order['Date'],
            buy_order['Symbol'],
            buy_order['Shares'],
            impact,
            commission,
            -1
        ),
        axis = 1
    )

    data_3.iloc[0, : -1] = data_2.iloc[0, : -1]
    data_3.iloc[0, -1] = data_3.iloc[0, -1] + data_2.iloc[0, -1]

    def edit_data_3_rows(df, df_2, index):
        df.iloc[index, : -1] = df_2.iloc[index, : -1] + df.iloc[index - 1, : -1]
        df.iloc[index, -1] = df_2.iloc[index, -1] + df.iloc[index - 1, -1]

    for index in np.arange(1, len(data_3)):
        edit_data_3_rows(data_3, data_2, index)

    port_vals = (data_1 * data_3).sum(axis = 1)
    return port_vals
