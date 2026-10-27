import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def run_historical_stresstest(df_risk):
    # Scale reference: compare cumulative peak-to-trough losses with scaled horizon VaR
    scenarios = {
        "COVID-19 Crash (24 Days)": ("2020-02-19", "2020-03-23", 24),
        "2022 Macro Rate Hike (251 Days)": ("2022-01-03", "2022-12-30", 251)
    }
    
    # Base daily 99% VaR observed inside the sample
    daily_var = df_risk['VaR_Historical'].mean()
    print("\n🔥 --- PHASE 4: HORIZON-ALIGNED STRESS TESTING ---")
    
    for name, (start, end, days) in scenarios.items():
        window_ret = df_risk['Realized_Return'].loc[start:end]
        cum_loss = np.exp(window_ret.sum()) - 1
        
        # Analytically scale the daily VaR using the square-root-of-time rule for a fair peer review
        scaled_var = daily_var * np.sqrt(days)
        print(f"\n📈 Scenario: {name}")
        print(f"   Realized Cumulative Loss:  {cum_loss*100:.2f}%")
        print(f"   Time-Scaled 99% VaR ({days}d): {scaled_var*100:.2f}%")
        if abs(cum_loss) > scaled_var:
            print("   ⚠️ Result: Loss EXCEEDS time-scaled VaR bounds (True non-linear tail event).")
        else:
            print("   ℹ️ Result: Loss falls WITHIN time-scaled VaR bounds (Horizon artifact corrected).")

def run_hypothetical_stresstest():
    """
    Simulates a comprehensive institutional macro shock vector incorporating:
    - Equity Tail Risk: -25%
    - Crypto Tail Volatility: -40%
    - Rates Shock + Credit Spreads: +200bps yield shift + 150bps credit spread widening
      TLT Price Impact (Duration=16.0, Convexity=3.2): dP = -D*dy + 0.5*C*(dy^2)
      LQD Price Impact (Duration=8.5, Spread Widening applies): dy_total = 2.0% + 1.5% = 3.5%
    """
    dy_rates = 0.02
    dy_lqd = 0.02 + 0.015  # Rates + Credit Spread
    
    tlt_shock = -16.0 * dy_rates + 0.5 * 3.2 * (dy_rates ** 2)
    lqd_shock = -8.5 * dy_lqd
    
    # Order matches config.TICKERS: ['SPY', 'IEUR', 'EEM', 'BTC-USD', 'TLT', 'LQD', 'GLD']
    shocks = np.array([-0.25, -0.25, -0.25, -0.40, tlt_shock, lqd_shock, 0.08])
    hypo_loss = np.sum(config.WEIGHTS * shocks)
    print(f"\n🔮 Hypothetical Severe Shock (Credit Spreads & Convexity Enabled): {hypo_loss*100:.2f}%")

if __name__ == "__main__":
    df = pd.read_csv(config.RISK_RESULTS_PATH, index_col=0, parse_dates=True)
    run_historical_stresstest(df)
    run_hypothetical_stresstest()


