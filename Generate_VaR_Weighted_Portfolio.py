import src.modules.util_get_data as utl_data
tickers = ['META','AMZN','NFLX','GOOG']
data = utl_data.get_data_yf(tickers)

close_values_selected = data['Close']

import plotly.graph_objects as go

# Assuming close_values_selected is a DataFrame containing the close values for selected tickers
fig = go.Figure()

for ticker in close_values_selected.columns:
    fig.add_trace(go.Scatter(x=close_values_selected.index, y=close_values_selected[ticker], mode='lines', name=ticker))

fig.update_layout(title='Stock Close Values',
                  xaxis_title='Date',
                  yaxis_title='Close Price')

fig.show()

returns = close_values_selected.pct_change()
import plotly.graph_objects as go

# Assuming close_values_selected is a DataFrame containing the close values for selected tickers
fig = go.Figure()

for ticker in returns.columns:
    fig.add_trace(go.Scatter(x=returns.index, y=returns[ticker], mode='lines', name=ticker))

fig.update_layout(title='Stock Returns',
                  xaxis_title='Date',
                  yaxis_title='Return')

fig.show()

import numpy as np

# Assuming returns is a DataFrame containing the returns for selected tickers
rolling_window = 30  # You can adjust this value according to your preference

# Calculate rolling standard deviation
rolling_volatility = returns.rolling(window=rolling_window).std()

# Adjust for annualization
rolling_volatility_annualized = rolling_volatility * np.sqrt(250)

# Drop rows with NaN values
rolling_volatility_annualized = rolling_volatility_annualized.dropna()

import plotly.graph_objects as go

# Assuming close_values_selected is a DataFrame containing the close values for selected tickers
fig = go.Figure()

for ticker in rolling_volatility_annualized.columns:
    fig.add_trace(go.Scatter(x=rolling_volatility_annualized.index, y=rolling_volatility_annualized[ticker], mode='lines', name=ticker))

fig.update_layout(title='Stock Rolling volatility',
                  xaxis_title='Date',
                  yaxis_title='Rolling Volatility')

fig.show()

# Assuming returns is a DataFrame containing the returns for selected tickers
rolling_window = 30  # You can adjust this value according to your preference
alpha = 0.9  # You can adjust this value according to your preference

# Calculate rolling standard deviation using exponential smoothing
rolling_volatility_ewma = returns.ewm(span=rolling_window, min_periods=rolling_window).std()

# Adjust for annualization
rolling_volatility_ewma_annualized = rolling_volatility_ewma * np.sqrt(250)

# Drop rows with NaN values
rolling_volatility_ewma_annualized = rolling_volatility_ewma_annualized.dropna()

import plotly.graph_objects as go

# Assuming close_values_selected is a DataFrame containing the close values for selected tickers
fig = go.Figure()

for ticker in rolling_volatility_ewma_annualized.columns:
    fig.add_trace(go.Scatter(x=rolling_volatility_ewma_annualized.index, y=rolling_volatility_ewma_annualized[ticker], mode='lines', name=ticker))

fig.update_layout(title='Stock Rolling volatility EWMA',
                  xaxis_title='Date',
                  yaxis_title='Rolling Volatility EWMA')

fig.show()

import scipy.stats as stats

# Assuming alpha = 0.05 (5% significance level)
alpha = 0.05
horizon_in_days = 250
# Calculate the z-score corresponding to the significance level
z_score = stats.norm.ppf(1 - alpha)

# Adjust for the horizon of 10 days and 250 trading days in a year
horizon_adjustment = np.sqrt(horizon_in_days/250)

# Calculate the 10-day VaR for each stock
var_10_day = rolling_volatility_annualized * z_score * horizon_adjustment

print(var_10_day)

import plotly.graph_objects as go

# Assuming close_values_selected is a DataFrame containing the close values for selected tickers
fig = go.Figure()

for ticker in var_10_day.columns:
    fig.add_trace(go.Scatter(x=var_10_day.index, y=var_10_day[ticker], mode='lines', name=ticker))

fig.update_layout(title='Stock var_10_day',
                  xaxis_title='Date',
                  yaxis_title='var_10_day')

fig.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming returns is a DataFrame containing the returns for selected tickers

# Calculate the correlation matrix
correlation_matrix = returns.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Returns')
plt.show()

# Assuming returns is a DataFrame containing the returns for the assets in the portfolio
# Assuming portfolio_weights is a list or array containing the weights of each asset in the portfolio
# Assuming var_level is the desired significance level (e.g., 0.05 for 5% VaR)
# Assuming horizon is the number of days in the horizon (e.g., 10 days)

import scipy.stats as stats

shares = np.array([100, 100, 100, 100]) 

portfolio_weights = np.array([0.25, 0.25, 0.25, 0.25])  # Example weights for a portfolio with 4 assets
var_level = 0.01  # 5% VaR
horizon = 250  # 10-day horizon

# Calculate the covariance matrix of returns
covariance_matrix = returns.cov()

# Adjust covariance matrix for the horizon factor
covariance_matrix *= horizon

# Calculate the portfolio variance
portfolio_variance = np.dot(portfolio_weights.T, np.dot(covariance_matrix, portfolio_weights))

# Calculate the portfolio standard deviation
portfolio_std_dev = np.sqrt(portfolio_variance)

# Calculate the z-score corresponding to the significance level
z_score = stats.norm.ppf(1 - var_level)

# Calculate the VaR of the portfolio
portfolio_var = z_score * portfolio_std_dev

print("Portfolio VaR ({}% confidence level, {}-day horizon): {:.2f}".format((1 - var_level) * 100, horizon, portfolio_var))





import scipy.stats as stats

var_level = 0.05  # 5% VaR
horizon = 250  # 10-day horizon
rolling_window_years = 2  # 2-year rolling window for covariance matrix calculation
shares = np.array([100, 100, 100, 100])  # Example number of shares for a portfolio with 4 assets

# Create an empty DataFrame to store rolling VaR
rolling_var = pd.DataFrame(index=returns.index)

# Calculate the number of trading days in the rolling window
trading_days_per_year = 250
rolling_window = trading_days_per_year * rolling_window_years

# Iterate over each rolling window
for i in np.arange(rolling_window, returns.shape[0]):
    # Slice the returns DataFrame to get the current rolling window for covariance matrix calculation
    returns_window_cov = returns.iloc[i-rolling_window:i]
    # Slice the returns DataFrame to get the current rolling window for VaR calculation
    returns_window_var = returns.iloc[i-rolling_window:i]

    # Calculate the covariance matrix of returns for the current window
    covariance_matrix = returns_window_cov.cov()
    
    # Calculate the portfolio weights for the current window
    prices = close_values_selected.iloc[i,:]
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
    z_score = stats.norm.ppf(1 - var_level)
    
    # Calculate the VaR of the portfolio for the current window
    portfolio_var = z_score * portfolio_std_dev
    
    # Store the rolling VaR in the DataFrame
    rolling_var.loc[returns_window_var.index[-1], 'Rolling_VaR'] = portfolio_var * np.sqrt(horizon)

# Drop NaN values
rolling_var = rolling_var.dropna()