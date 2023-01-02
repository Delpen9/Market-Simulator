import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data

from marketsimcode import compute_portfolio_values

class TheoreticallyOptimalStrategy:
	def testPolicy(
        self,
        symbols : list,
        sd : dt.datetime,
        ed : dt.datetime,
        sv : int
    ) -> any:
            """
            Compute the results of the theoretically optimal strategy as an upper bound on performance;
            use this as a comparison benchmark.

            :param symbols : Symbol to use in analyzing technical indicator
            :type symbols: list

            :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
            :type start_date: datetime

            :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
            :type end_date: datetime

            :param starting_value: Int object representing starting value of the portfolio
            :type starting_value : int
            """
            dates = pd.date_range(sd, ed)
            prices = get_data(symbols, dates)[symbols].ffill().bfill()

            trades = prices.copy().drop(symbols, axis = 1, inplace = False)
            trades['Shares'] = 0.0

            trades['Symbol'] = symbols[0]
            trades['Date'] = trades.index

            net_holdings = 0.0

            dates = prices.index
            for i in range(len(dates) - 1):
                if prices.loc[dates[i], symbols[0]] < prices.loc[dates[i + 1], symbols[0]]:
                    new_trade = 1000 - net_holdings
                else:
                    new_trade = -1000 - net_holdings

                net_holdings += new_trade

                trades.loc[dates[i], 'Shares'] = abs(new_trade)

                if new_trade < 0:
                    trades.loc[dates[i], 'Order'] = 'SELL'
                else:
                    trades.loc[dates[i], 'Order'] = 'BUY'

            trades = trades.reindex(columns = ['Date', 'Symbol', 'Order', 'Shares']).reset_index(drop = True)
            trades['Date'] = trades['Date'].values.astype(str)
            trades['Date'] = trades.apply(lambda row : row['Date'].split('T')[0], axis = 1)
            return trades

def benchmark(
    symbols : list,
    sd : dt.datetime,
    ed : dt.datetime,
    sv : int
) -> any:
        """
        Compute the benchmark values. This refers to buying 1000 shares at the beginning of the year then selling those shares at the end.

        :param symbols : Symbol to use in analyzing technical indicator
        :type symbols: list

        :param start_date: A datetime object that represents the start date, defaults to 1/1/2008
        :type start_date: datetime

        :param end_date: A datetime object that represents the end date, defaults to 1/1/2009
        :type end_date: datetime

        :param starting_value: Int object representing starting value of the portfolio
        :type starting_value : int
        """
        dates = pd.date_range(sd, ed)
        prices = get_data(symbols, dates)[symbols].ffill().bfill()
  
        trades = prices.copy().drop(symbols, axis = 1, inplace = False).iloc[ :2]

        trades['Date'] = [str(sd + dt.timedelta(days = 1)).split(' ')[0], str(ed).split(' ')[0]]
        trades['Symbol'] = [symbols[0], symbols[0]]
        trades['Order'] = ['BUY', 'SELL']
        trades['Shares'] = [1000, 1000]

        portvals = compute_portfolio_values(trades, sv, commission = 0.00, impact = 0.00)
        return portvals

def plot_data(df, title = '', xlabel = '', ylabel = '', name = 'Figure_1.png'):
    ax = df.plot(title = title, fontsize = 9, color = ['purple', 'red'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(plt.MaxNLocator(12))
    plt.savefig(name)
    plt.clf()

def generate_plots(
    benchmark_values : pd.DataFrame,
    theoretically_optimal_values : pd.DataFrame
) -> None:
    benchmark = pd.DataFrame(benchmark_values.values, index = benchmark_values.index, columns = ['benchmark'])
    TOS = pd.DataFrame(theoretically_optimal_values.values, index = theoretically_optimal_values.index, columns = ['theoretically_optimal_strategy'])
    prices = pd.merge(benchmark, TOS, left_index = True, right_index = True)
    prices = prices.apply(lambda row : row / prices.iloc[0], axis = 1)
    plot_data(prices, title = fr'Benchmark VS. Theoretically Optimal Strategy', xlabel = 'Date', ylabel = 'Normalized Price', name = 'Figure_11.png')

def table(
    benchmark_values : pd.DataFrame,
    theoretically_optimal_values : pd.DataFrame
) -> None:
    benchmark = pd.DataFrame(benchmark_values.values, index = benchmark_values.index, columns = ['benchmark'])
    theoretical = pd.DataFrame(theoretically_optimal_values.values, index = theoretically_optimal_values.index, columns = ['theoretically_optimal_strategy'])

    cumulative_return_of_benchmark = (benchmark.iloc[-1] - benchmark.iloc[0]).values[0]
    cumulative_return_of_TOS = (theoretical.iloc[-1] - theoretical.iloc[0]).values[0]

    daily_return_of_benchmark = benchmark.pct_change(periods = 1) * benchmark.shift(1)
    daily_return_of_TOS = theoretical.pct_change(periods = 1) * theoretical.shift(1)

    standard_deviation_of_daily_returns_benchmark = daily_return_of_benchmark.std().values[0]
    standard_deviation_of_daily_returns_theoretically_optimal_strategy = daily_return_of_TOS.std().values[0]

    mean_of_daily_returns_benchmark = daily_return_of_benchmark.mean().values[0]
    mean_of_daily_returns_theoretically_optimal_strategy = daily_return_of_TOS.mean().values[0]

    data = [
        ['Cumulative Return\n of TOS', '{:.6f}'.format(cumulative_return_of_TOS)],
        ['Standard Deviation\n of Daily Return of TOS', '{:.6f}'.format(standard_deviation_of_daily_returns_theoretically_optimal_strategy)],
        ['Mean Daily Return\n of TOS', '{:.6f}'.format(mean_of_daily_returns_theoretically_optimal_strategy)],
        ['Cumulative Return\n of Benchmark', '{:.6f}'.format(cumulative_return_of_benchmark)],
        ['Standard Deviation\n of Daily Return of Benchmark', '{:.6f}'.format(standard_deviation_of_daily_returns_benchmark)],
        ['Mean Daily Return\n of Benchmark','{:.6f}'.format( mean_of_daily_returns_benchmark)],
    ]

    fig, ax = plt.subplots()
    table = ax.table(cellText = data, cellLoc = 'center', loc = 'center')
    fig.tight_layout()
    table.set_fontsize(14)
    table.scale(1, 4)
    ax.axis('off')
    plt.savefig('Figure_12.png')
    plt.clf()

def run_theoretically_optimal_strategy():
    symbols = ['JPM']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000

    TOS = TheoreticallyOptimalStrategy()
    trades = TOS.testPolicy(symbols, sd = start_date, ed = end_date, sv = starting_value)
    theoretically_optimal_strategy_values = compute_portfolio_values(
        trades,
        starting_value,
        commission = 0.00,
        impact = 0.00
    )

    benchmark_values = benchmark(symbols, start_date, end_date, starting_value)

    generate_plots(benchmark_values, theoretically_optimal_strategy_values)
    table(benchmark_values, theoretically_optimal_strategy_values)

if __name__ == "__main__":
	run_theoretically_optimal_strategy()