#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.stats as stats





def get_returns(df:pd.DataFrame)->pd.DataFrame:
    """
    Calculate the percentage change in values for each column of the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing numerical data.

    Returns:
    - pd.DataFrame: DataFrame containing the percentage change in values for each column.
      The first row will contain NaN values since there's no prior row to compare with.
    """
    returns = df.pct_change()
    return returns

def get_rolling_volatility_from_daily_values(df:pd.DataFrame, 
                                      rolling_window = 30, nb_trading_days=250)->pd.DataFrame:
    """
    Calculate the rolling annualized volatility from daily values.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing daily values.
    - rolling_window (int): Window size for rolling calculation (default: 30).
    - nb_trading_days (int): Number of trading days in a year for annualization (default: 250).

    Returns:
    - pd.DataFrame: DataFrame containing the rolling annualized volatility for each column.
    """
    # Assuming returns is a DataFrame containing the returns for selected tickers
    
    # Calculate rolling standard deviation
    rolling_volatility = df.rolling(window=rolling_window).std()
    
    # Adjust for annualization
    rolling_volatility_annualized = rolling_volatility * np.sqrt(nb_trading_days)
    
    # Drop rows with NaN values
    rolling_volatility_annualized = rolling_volatility_annualized.dropna()

    return rolling_volatility_annualized

def get_rolling_volatility_from_daily_values_with_ewma(df:pd.DataFrame, 
                                      rolling_window = 30, 
                                      nb_trading_days=250,
                                      alpha = 0.9)->pd.DataFrame:
    """
    Calculate the rolling annualized volatility from daily values using Exponentially Weighted Moving Average (EWMA).

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing daily values.
    - rolling_window (int): Window size for EWMA calculation (default: 30).
    - nb_trading_days (int): Number of trading days in a year for annualization (default: 250).
    - alpha (float): Smoothing factor for EWMA (default: 0.9).

    Returns:
    - pd.DataFrame: DataFrame containing the rolling annualized volatility calculated using EWMA for each column.
    """
    
    # Calculate rolling standard deviation using exponential smoothing
    rolling_volatility_ewma = df.ewm(span=rolling_window, min_periods=rolling_window).std()
    
    # Adjust for annualization
    rolling_volatility_ewma_annualized = rolling_volatility_ewma * np.sqrt(250)
    
    # Drop rows with NaN values
    rolling_volatility_ewma_annualized = rolling_volatility_ewma_annualized.dropna()

    return rolling_volatility_ewma_annualized


def get_hday_rolling_var_from_daily_values(data:pd.DataFrame,
                                            rolling_window=30,
                                            nb_trading_days=250,
                                            alpha=0.05,
                                            horizon_in_days=250):
    """
    Calculate the rolling Value at Risk (VaR) for a given horizon using historical simulation.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing daily values.
    - rolling_window (int): Window size for rolling calculation (default: 30).
    - nb_trading_days (int): Number of trading days in a year for annualization (default: 250).
    - alpha (float): Significance level for VaR calculation (default: 0.05).
    - horizon_in_days (int): Horizon for VaR calculation (default: 250).

    Returns:
    - pd.DataFrame: DataFrame containing the rolling VaR for each column.
    """
    returns = get_returns(data)
    rolling_volatility_annualized = get_rolling_volatility_from_daily_values(returns,
                                                                                            nb_trading_days=nb_trading_days,
                                                                                            rolling_window=rolling_window)
    # Calculate the z-score corresponding to the significance level
    z_score = stats.norm.ppf(1 - alpha)
    
    # Adjust for the horizon of 10 days and 250 trading days in a year
    horizon_adjustment = np.sqrt(horizon_in_days/nb_trading_days)
    
    # Calculate the 10-day VaR for each stock
    var_h_day = rolling_volatility_annualized * z_score * horizon_adjustment

    return var_h_day
    