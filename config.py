import numpy as np

# Portfolio Configuration
TICKERS = ['SPY', 'IEUR', 'EEM', 'BTC-USD', 'TLT', 'LQD', 'GLD']
WEIGHTS = np.array([0.30, 0.15, 0.10, 0.05, 0.15, 0.15, 0.10])

# Time Horizons
START_DATE = "2007-01-01"  # Data ingestion starts here (BTC will limit effective start to Sep 2014)
WINDOW_SIZE = 500          # Rolling window days
CONFIDENCE_LEVEL = 0.99
RANDOM_SEED = 42
MC_SIMULATIONS = 20000     # Increased to reduce quantile noise at 99%

# File Paths
RAW_DATA_PATH = "data/raw_prices.csv"
RISK_RESULTS_PATH = "data/rolling_risk_results.csv"
PLOT_DIR = "plots/"
