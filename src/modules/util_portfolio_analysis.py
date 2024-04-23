#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
import scipy.stats as stats


def normalize_portfolio_weights(portfolio_description_raw:dict)->dict:
    """
    Normalize the weights of assets in a portfolio description.

    This function takes a dictionary containing the description of a portfolio,
    where each key represents an asset and the corresponding value is its weight.
    It normalizes the weights so that they sum up to 1.

    Parameters:
    - portfolio_description_raw (dict): Dictionary containing the description of the portfolio.
      The dictionary should have the following structure:
      {
          "weights": {
              "asset1": weight1,
              "asset2": weight2,
              ...
          }
      }

    Returns:
    - dict: Normalized portfolio description where the weights of assets sum up to 1.
      The structure of the returned dictionary is the same as the input dictionary.
    """
    portfolio_description = copy.deepcopy(portfolio_description_raw)
    initial_weights_sum = np.sum(list(portfolio_description["weights"].values()))
    for i in portfolio_description["weights"]:
        portfolio_description["weights"][i] = portfolio_description["weights"][i]/initial_weights_sum
    return portfolio_description


def get_linear_portfolio_var_based_on_weights(returns,
                                              portfolio_description_raw,
                                              alpha=0.05,
                                              nb_days_for_to_estimate = 500,
                                              horizon = 250)->float:
    """
    Calculate the linear Value at Risk (VaR) of a portfolio based on specified weights.

    This function calculates the linear VaR of a portfolio using historical simulation.
    It takes the returns of assets, a description of the portfolio including asset weights,
    and additional parameters such as confidence level, number of days for estimation, and horizon.

    Parameters:
    - returns (pd.DataFrame): DataFrame containing the returns of assets.
    - portfolio_description_raw (dict): Dictionary containing the description of the portfolio.
      The dictionary should have the following structure:
      {
          "weights": {
              "asset1": weight1,
              "asset2": weight2,
              ...
          }
      }
    - alpha (float): Significance level for VaR calculation (default: 0.05).
    - nb_days_for_to_estimate (int): Number of days to use for covariance estimation (default: 500).
    - horizon (int): Horizon for VaR calculation (default: 250).

    Returns:
    - float: Linear VaR of the portfolio.
    """
    tickers = returns.columns
    portfolio_description = normalize_portfolio_weights(portfolio_description_raw)
    portfolio_weights = np.array([portfolio_description["weights"][i] for i in tickers])
    
    # Calculate the covariance matrix of returns
    covariance_matrix = returns[-nb_days_for_to_estimate:].cov()
    
    # Adjust covariance matrix for the horizon factor
    covariance_matrix *= horizon
    
    # Calculate the portfolio variance
    portfolio_variance = np.dot(portfolio_weights.T, np.dot(covariance_matrix, portfolio_weights))
    
    # Calculate the portfolio standard deviation
    portfolio_std_dev = np.sqrt(portfolio_variance)
    
    # Calculate the z-score corresponding to the significance level
    z_score = stats.norm.ppf(1 - alpha)
    
    # Calculate the VaR of the portfolio
    portfolio_var = z_score * portfolio_std_dev
    
    print("Portfolio VaR ({}% confidence level, {}-day horizon): {:.2f}".format((1 - alpha) * 100, horizon, portfolio_var))

    return portfolio_var


def get_rolling_linear_portfolio_var_based_on_weights(returns,
                                                      portfolio_description,
                                                      data,
                                                      stock_values=None,
                                                      alpha = 0.05,
                                                      horizon = 250,
                                                      rolling_window_in_days = 500):
    
    portfolio_description = normalize_portfolio_weights(portfolio_description)
        
    if stock_values is None:
        stock_values = data.loc[data.index.max()]
    portions = {ticker: portfolio_description['value'] * weight for ticker, weight in portfolio_description['weights'].items()}
    shares_dict = {ticker: portion / stock_values[ticker] for ticker, portion in portions.items()}
    tickers = returns.columns
    shares = np.array([shares_dict[i] for i in tickers])
    
    # Create an empty DataFrame to store rolling VaR
    rolling_var = pd.DataFrame(index=returns.index)
    
    # Iterate over each rolling window
    for i in np.arange(rolling_window_in_days, returns.shape[0]):
        # Slice the returns DataFrame to get the current rolling window for covariance matrix calculation
        returns_window_cov = returns.iloc[i-rolling_window_in_days:i]
        # Slice the returns DataFrame to get the current rolling window for VaR calculation
        returns_window_var = returns.iloc[i-rolling_window_in_days:i]
    
        # Calculate the covariance matrix of returns for the current window
        covariance_matrix = returns_window_cov.cov()
        
        # Calculate the portfolio weights for the current window
        prices = data.iloc[i,:]
        market_values = prices * shares
        # Calculate the total market value of the portfolio
        total_market_value = market_values.sum()
        # Calculate the weights of each asset
        weights = market_values / total_market_value
    
        # Calculate the portfolio variance for the current window
        portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
        
        # Calculate the portfolio standard deviation for the current window
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        # Calculate the z-score corresponding to the significance level
        z_score = stats.norm.ppf(1 - alpha)
        
        # Calculate the VaR of the portfolio for the current window
        portfolio_var = z_score * portfolio_std_dev
        
        # Store the rolling VaR in the DataFrame
        rolling_var.loc[returns_window_var.index[-1], 'Rolling_VaR'] = portfolio_var * np.sqrt(horizon)
    
    # Drop NaN values
    rolling_var = rolling_var.dropna()

    return rolling_var