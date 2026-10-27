import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def calculate_advanced_stats(portfolio_returns):
    if isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = portfolio_returns.index.tz_localize(None)

    skew = stats.skew(portfolio_returns)
    kurt = stats.kurtosis(portfolio_returns)  # Excess Kurtosis
    jb_stat, jb_pvalue = stats.jarque_bera(portfolio_returns)
    
    print("\n📊 --- EXPLORATORY STATISTICAL MOMENTS (Full Sample) ---")
    print(f"Sample Period:            {portfolio_returns.index[0].strftime('%Y-%m-%d')} to {portfolio_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Skewness (Asymmetry):     {skew:.4f}")
    print(f"Kurtosis (Excess):        {kurt:.4f}")
    print(f"Jarque-Bera p-value:      {jb_pvalue:.6f}")
    return skew, kurt

def plot_and_save_eda(portfolio_returns, asset_returns):
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    if isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = portfolio_returns.index.tz_localize(None)
        asset_returns.index = asset_returns.index.tz_localize(None)

    # 1. Normality Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(portfolio_returns, kde=False, stat="density", bins=50, color="darkblue", alpha=0.6, ax=ax1)
    mu, std = portfolio_returns.mean(), portfolio_returns.std()
    x = np.linspace(mu - 4*std, mu + 4*std, 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), color="red", lw=2, label="Theoretical Gaussian")
    ax1.set_title("Returns Distribution vs Gaussian")
    ax1.legend()
    
    stats.probplot(portfolio_returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Tail Deviations)")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "normality_check.png"))
    plt.close()

    # 2. Correlation Breakdowns (Verify the true SPY/TLT daily returns in 2022)
    calm = asset_returns.loc["2017-01-01":"2019-12-31"]
    y22  = asset_returns.loc["2022-01-01":"2022-12-31"]
    
    a = calm["SPY"].corr(calm["TLT"])
    b = y22["SPY"].corr(y22["TLT"])
    print(f"\n🗺️ Verification: Daily SPY/TLT Corr | Calm: {a:.2f} | 2022 Rate Crisis: {b:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(calm.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Calm Regime (2017-2019)")
    sns.heatmap(y22.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1], cbar=False)
    axes[1].set_title("2022 Rate Hike Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "correlation_regimes.png"))
    plt.close()

if __name__ == "__main__":
    from data_loader import fetch_and_freeze_data, get_portfolio_returns
    df = fetch_and_freeze_data()
    a_ret, p_ret = get_portfolio_returns(df)
    calculate_advanced_stats(p_ret)
    plot_and_save_eda(p_ret, a_ret)



