import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data

def plot_data(df, title = '', xlabel = '', ylabel = '', name = 'Figure_1.png'):
    ax = df.plot(title = title, fontsize = 9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.savefig(name)
    plt.clf()

def bollinger_bands(
    start_date : dt.datetime,
    end_date : dt.datetime,
    symbols : list,
    m : int = 20,
    figure_name : str = None,
    gen_plot : bool = False
) -> None:
    """
    This function creates an upper and lower bollinger band following a stock's price in a specified date range.

    :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
    :type start_date: datetime

    :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
    :type end_date: datetime

    :param symbols : Symbol to use in analyzing technical indicator
    :type syms: list

    :param m : Number of days to include in moving average
    :type syms: int

    :param figure_name : Name of output figure in gen_plot
    :type figure_name: str

    :param gen_plot: If True, optionally saves a plot.
    :type gen_plot: bool
    """
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)[symbols]
    prices['std'] = prices.rolling(m).std(ddof = 0)
    prices['lower'] = prices.apply(lambda row : row[symbols] - 2 * row['std'], axis = 1)
    prices['upper'] = prices.apply(lambda row : row[symbols] + 2 * row['std'], axis = 1)  
    prices.drop('std', axis = 1, inplace = True)

    if gen_plot:
        plot_data(prices, title = fr'Bollinger Bands of {m} Day Moving Average', xlabel = 'Date', ylabel = 'Price', name = figure_name)

def exponential_moving_average(
    start_date : dt.datetime,
    end_date : dt.datetime,
    symbols : list,
    m : int = 20,
    figure_name : str = None,
    gen_plot : bool = False
) -> None:
    """
    This function calculates the exponential moving average of a stock price.

    :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
    :type start_date: datetime

    :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
    :type end_date: datetime

    :param symbols : Symbol to use in analyzing technical indicator
    :type syms: list

    :param m : Number of days to include in moving average
    :type syms: int

    :param figure_name : Name of output figure in gen_plot
    :type figure_name: str

    :param gen_plot: If True, optionally saves a plot.
    :type gen_plot: bool
    """
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)[symbols]

    k = 2 / (m + 1)

    prices['ema'] = prices[symbols]
    prices.reset_index(drop = True, inplace = True)
    for i in range(1, len(prices)):
        prices.loc[i, 'ema'] = prices.loc[i, symbols[0]] * k + prices.loc[i - 1, 'ema'] * (1 - k)

    if gen_plot:
        plot_data(prices, title = fr'Exponential Moving Average of {m} Day Moving Average', xlabel = 'Date', ylabel = 'Price', name = figure_name)

def rate_of_change(
    start_date : dt.datetime,
    end_date : dt.datetime,
    symbols : list,
    m : int = 20,
    figure_name : str = None,
    gen_plot : bool = False
) -> None:
    """
    This function calculates the % rate of change given a time delta, m.

    :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
    :type start_date: datetime

    :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
    :type end_date: datetime

    :param symbols : Symbol to use in analyzing technical indicator
    :type syms: list

    :param m : Number of days to include in moving average
    :type syms: int

    :param figure_name : Name of output figure in gen_plot
    :type figure_name: str

    :param gen_plot: If True, optionally saves a plot.
    :type gen_plot: bool
    """
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)[symbols]

    prices['price_change'] = prices[symbols[0]].pct_change(periods = m)

    if gen_plot:
        plot_data(prices['price_change'], title = fr'{m} Day Rate of Change', xlabel = 'Date', ylabel = fr'% Change per {m} Days', name = figure_name)

def stochastic_oscillator_indicator(
    start_date : dt.datetime,
    end_date : dt.datetime,
    symbols : list,
    m : int = 20,
    n : int = 5,
    figure_name : str = None,
    gen_plot : bool = False
) -> None:
    """
    This function calculates the stochastic oscillator indicator of a standard moving average.
    An oscillator value above 80% implies that the stock is overbought,
    while an oscillator value of below 20% implies that the stock is underbought.

    :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
    :type start_date: datetime

    :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
    :type end_date: datetime

    :param symbols : Symbol to use in analyzing technical indicator
    :type syms: list

    :param m : Number of days to include in price min/max calculation.
    :type syms: int

    :param n : Number of days to include in moving average
    :type syms: int

    :param figure_name : Name of output figure in gen_plot
    :type figure_name: str

    :param gen_plot: If True, optionally saves a plot.
    :type gen_plot: bool
    """
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)[symbols]

    prices['high'] = prices[symbols[0]].rolling(m).max()
    prices['low'] = prices[symbols[0]].rolling(m).min()
    prices['%K'] = (prices[symbols[0]] - prices['low']) * 100 / (prices['high'] - prices['low'])
    prices['%D'] = prices['%K'].rolling(n).mean()

    if gen_plot:
        plot_data(prices['%D'], title = fr'Stochastic Oscillator Indicator for {m} Days on {n} Day Moving Average', xlabel = 'Date', ylabel = fr'%', name = figure_name)

def commodity_channel_index(
    start_date : dt.datetime,
    end_date : dt.datetime,
    symbols : list,
    m : int = 20,
    figure_name : str = None,
    gen_plot : bool = False
) -> None:
    """
    This function calculates the commodity channel index of price data for a stock.

    :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
    :type start_date: datetime

    :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
    :type end_date: datetime

    :param symbols : Symbol to use in analyzing technical indicator
    :type syms: list

    :param m : Number of days to include in moving average
    :type syms: int

    :param figure_name : Name of output figure in gen_plot
    :type figure_name: str

    :param gen_plot: If True, optionally saves a plot.
    :type gen_plot: bool
    """
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)[symbols]

    lamberts_constant = 0.015

    prices['sma'] = prices[symbols].rolling(m).mean()
    prices['std'] = prices[symbols].rolling(m).std(ddof = 0)
    prices['asd'] = prices['std'].mean()
    prices['cci'] = prices.apply(lambda row : (row[symbols] - row['sma']) / lamberts_constant * row['asd'], axis = 1)

    if gen_plot:
        plot_data(prices['cci'], title = fr'{m} Day Commodity Channel Index', xlabel = 'Date', ylabel = fr'Index', name = figure_name)

def run_indicators():
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    symbols = ['JPM']
    bollinger_bands(start_date, end_date, symbols, m = 20, figure_name = 'Figure_1.png', gen_plot = True)
    bollinger_bands(start_date, end_date, symbols, m = 100, figure_name = 'Figure_2.png', gen_plot = True)
    exponential_moving_average(start_date, end_date, symbols, m = 10, figure_name = 'Figure_3.png', gen_plot = True)
    exponential_moving_average(start_date, end_date, symbols, m = 100, figure_name = 'Figure_4.png', gen_plot = True)
    rate_of_change(start_date, end_date, symbols, m = 10, figure_name = 'Figure_5.png', gen_plot = True)
    rate_of_change(start_date, end_date, symbols, m = 100, figure_name = 'Figure_6.png', gen_plot = True)
    stochastic_oscillator_indicator(start_date, end_date, symbols, m = 20, n = 5, figure_name = 'Figure_7.png', gen_plot = True)
    stochastic_oscillator_indicator(start_date, end_date, symbols, m = 100, n = 5, figure_name = 'Figure_8.png', gen_plot = True)
    commodity_channel_index(start_date, end_date, symbols, m = 20, figure_name = 'Figure_9.png', gen_plot = True)
    commodity_channel_index(start_date, end_date, symbols, m = 100, figure_name = 'Figure_10.png', gen_plot = True)

if __name__ == "__main__":
	run_indicators()