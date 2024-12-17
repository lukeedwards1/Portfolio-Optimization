import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
import matplotlib.pyplot as plt
from pypfopt import plotting
from datetime import datetime
from matplotlib.ticker import FuncFormatter

################################################# Collecting Data #########################################################

# Validate date input
def validate_date(prompt):
    while True:
        try:
            date_str = input(prompt)
            datetime.strptime(date_str, "%Y-%m-%d")
            return date_str
        except ValueError:
            print("Invalid date format. Please enter date as YYYY-MM-DD.")

# Validate user input for tickers
while True:
    try:
        tickers = [ticker.strip().upper() for ticker in input("Enter stock tickers separated by commas (e.g., AAPL, MSFT, AMZN, GOOG): ").split(',')]
        if all(t.isalpha() for t in tickers):  # Ensures only letters are entered
            break
        else:
            print("Invalid ticker input. Please enter valid stock symbols (letters only).")
    except Exception as e:
        print(f"Error: {e}. Try again.")

# Collect start and end dates
start_date = validate_date("Enter start date (YYYY-MM-DD): ")
end_date = validate_date("Enter end date (YYYY-MM-DD): ")

print(f"\nDownloading data for: {', '.join(tickers)} from {start_date} to {end_date}...")
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Check for missing data
if data.empty or data.isnull().all().all():
    print("\nNo valid data retrieved. Check the tickers or date range.")
    exit()

# Save data locally
data.to_csv('stock_prices.csv')
print(data.head())

################################################# Calculating Returns and Risk #############################################

# Load stock price data
data = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("\nMean Returns:\n", mean_returns)
print("\nCovariance Matrix:\n", cov_matrix)

################################################# Portfolio Optimization ###################################################

# Plot the Efficient Frontier
ef_plot = EfficientFrontier(mean_returns, cov_matrix)
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

# Add stock labels to the black dots (below each point)
for i, ticker in enumerate(data.columns):
    ret = mean_returns[ticker]
    vol = np.sqrt(cov_matrix.loc[ticker, ticker])
    ax.text(vol, ret - 0.00002, ticker, fontsize=10, ha='center', va='top', color='black')

# Highlight the Max Sharpe Ratio Portfolio point
ef = EfficientFrontier(mean_returns, cov_matrix)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
ret, vol, _ = ef.portfolio_performance()
max_sharpe = ax.scatter(vol, ret, c="red", marker="*", s=100, label="Max Sharpe Ratio")

# Highlight the Minimum Volatility Portfolio point
ef_min_vol = EfficientFrontier(mean_returns, cov_matrix)
min_vol_weights = ef_min_vol.min_volatility()
clean_min_vol_weights = ef_min_vol.clean_weights()
min_vol_ret, min_vol_vol, _ = ef_min_vol.portfolio_performance()
min_vol = ax.scatter(min_vol_vol, min_vol_ret, c="green", marker="*", s=100, label="Min Volatility Portfolio")

# Highlight the Equal-Weighted Portfolio point
equal_weights = np.array([1 / len(data.columns)] * len(data.columns))
equal_return = np.dot(mean_returns, equal_weights)
equal_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
equal_portfolio = ax.scatter(equal_volatility, equal_return, c="blue", marker="*", s=100, label="Equal-Weighted Portfolio")


# Add legend manually with unique handles
ax.legend(
    handles=[
        plt.Line2D([], [], color="blue", label="Efficient Frontier"),
        plt.Line2D([], [], marker='o', color='black', linestyle='None', markersize=8, label="Assets"),
        plt.Line2D([], [], marker='*', color='red', linestyle='None', markersize=12, label="Max Sharpe Ratio"),
        plt.Line2D([], [], marker='*', color='green', linestyle='None', markersize=12, label="Min Volatility Portfolio"),
        plt.Line2D([], [], marker='*', color='blue', linestyle='None', markersize=12, label="Equal-Weighted Portfolio")
    ],
    loc="upper left"
)


# Finalize plot
plt.title("Efficient Frontier with Portfolio Assets and Labels")
plt.show()

################################################# Backtesting Portfolio Performance ########################################

print("\nBacktesting Portfolios...")
optimized_weights = np.array([cleaned_weights[ticker] for ticker in tickers])
optimized_daily_returns = returns @ optimized_weights
equal_daily_returns = returns @ equal_weights

optimized_cumulative = (1 + optimized_daily_returns).cumprod()
equal_cumulative = (1 + equal_daily_returns).cumprod()

# Plot cumulative returns with percentages on y-axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(optimized_cumulative, label="Optimized Portfolio (Max Sharpe)", color='red')
ax.plot(equal_cumulative, label="Equal-Weighted Portfolio", color='blue')

# Set y-axis ticks and format as percentages
ax.set_ylim(0, None)  # Start y-axis at 0
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}%"))  # One decimal precision
ax.yaxis.set_ticks(np.arange(0, ax.get_ylim()[1] + 1, 0.5))  # Ticks every 0.5%


plt.title("Backtest: Cumulative Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (%)")
plt.legend()
plt.show()

################################################# Saving Results ###########################################################

# Combine all results into a single DataFrame
results = pd.DataFrame({
    "Optimized Portfolio (Max Sharpe)": [cleaned_weights.get(ticker, 0) for ticker in tickers],
    "Minimum Volatility Portfolio": [clean_min_vol_weights.get(ticker, 0) for ticker in tickers],
    "Equal-Weighted Portfolio": equal_weights
}, index=tickers)

# Save results to CSV
results.to_csv('portfolio_results.csv')
print("\nPortfolio results saved to 'portfolio_results.csv'.")
