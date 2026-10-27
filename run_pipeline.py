import os
from sources.data_loader import fetch_and_freeze_data, get_portfolio_returns
from sources.eda_analysis import calculate_advanced_stats, plot_and_save_eda
from sources.metrics import generate_rolling_risk_metrics
from sources.backtesting import execute_backtesting_suite
from sources.stress_testing import run_historical_stresstest, run_hypothetical_stresstest

def main():
    print("========== STARTING AUTOMATED QUANT RISK PIPELINE ==========")
    prices = fetch_and_freeze_data()
    asset_ret, port_ret = get_portfolio_returns(prices)
    
    calculate_advanced_stats(port_ret)
    plot_and_save_eda(port_ret, asset_ret)
    
    generate_rolling_risk_metrics(asset_ret, port_ret)
    execute_backtesting_suite()
    
    import pandas as pd
    import config
    df_results = pd.read_csv(config.RISK_RESULTS_PATH, index_col=0, parse_dates=True)
    run_historical_stresstest(df_results)
    run_hypothetical_stresstest()
    print("========== PIPELINE EXECUTED SUCCESSFULLY AND IMAGES SAVED ==========")

if __name__ == "__main__":
    main()
