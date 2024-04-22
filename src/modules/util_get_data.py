#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yfinance as yf
from datetime import datetime



def get_data_yf(tickers, start="2000-01-01", end=None, period="1d", output_pkl="./data/yf_stocks_values.pkl"):
    """
    Download historical stock data from Yahoo Finance for the specified tickers.

    Parameters:
        tickers (str or list): A single ticker symbol or a list of ticker symbols.
        start (str): Start date for the historical data in "YYYY-MM-DD" format. Default is None.
        end (str or None): End date for the historical data in "YYYY-MM-DD" format. Default is "2024-04-22".
                           If set to None, end date will be set to today's date.
        period (str): Interval between data points. Valid values are "1d" (daily), "1wk" (weekly), or "1mo" (monthly). Default is "1d".
        output_pkl (str): Path to save the downloaded data as a pickle file. Default is "./data/yf_stocks_values.pkl". Set to None to return data without saving.

    Returns:
        pandas.DataFrame or dict: If output_pkl is None, returns a pandas DataFrame containing the downloaded data. 
                                  Otherwise, saves the data as a pickle file and returns the same DataFrame.
    """
    if end is None:
        today_date = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(tickers, start=start, end=today_date, period=period)
    else:
        data = yf.download(tickers, start=start, end=end, period=period)

    if output_pkl is None:
        return data
    data.to_pickle(output_pkl)
    return data