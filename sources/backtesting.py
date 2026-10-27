import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

# Correzione del path per caricare config.py dalla cartella principale
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def run_kupiec_test(violations, total_days, expected_ratio=0.01):
    observed_ratio = violations / total_days
    if violations == 0 or violations == total_days:
        return 0.0, 1.0
    lr_part1 = (total_days - violations) * np.log((1 - observed_ratio) / (1 - expected_ratio))
    lr_part2 = violations * np.log(observed_ratio / expected_ratio)
    lr_stat = 2 * (lr_part1 + lr_part2)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return lr_stat, p_value

def run_christoffersen_independence(violation_series):
    v = np.array(violation_series)
    n = len(v)
    n00 = np.sum((v[:-1] == 0) & (v[1:] == 0))
    n01 = np.sum((v[:-1] == 0) & (v[1:] == 1))
    n10 = np.sum((v[:-1] == 1) & (v[1:] == 0))
    n11 = np.sum((v[:-1] == 1) & (v[1:] == 1))
    
    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p = (n01 + n11) / (n - 1) if (n - 1) > 0 else 0
    
    ln_l0 = (n00 + n10) * np.log(1 - p + 1e-10) + (n01 + n11) * np.log(p + 1e-10)
    ln_la = n00 * np.log(1 - p01 + 1e-10) + n01 * np.log(p01 + 1e-10) + \
            n10 * np.log(1 - p11 + 1e-10) + n11 * np.log(p11 + 1e-10)
            
    lr_stat = 2 * (ln_la - ln_l0)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    return lr_stat, p_value

def get_basel_zone(violations, total_days):
    scaled_violations = violations * (250 / total_days)
    if scaled_violations <= 4.0:
        return "GREEN 🟢"
    elif scaled_violations <= 9.0:
        return "YELLOW 🟡"
    return "RED 🔴"

def execute_backtesting_suite():
    # Carica i dati di rischio generati da metrics.py
    df = pd.read_csv(config.RISK_RESULTS_PATH, index_col=0, parse_dates=True)
    total_days = len(df)
    alpha = 1 - config.CONFIDENCE_LEVEL
    models = ['VaR_Historical', 'VaR_Param_Static', 'VaR_Param_EWMA', 'VaR_MC_Normal', 'VaR_MC_Student']
    
    print(f"\n📊 --- BACKTESTING REPORT (Window: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}) ---")
    print(f"Backtesting Days: {total_days} | Expected Violations: {total_days * alpha:.2f}")
    
    summary = []
    # Crea automaticamente anche la cartella per i grafici se non esiste
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    
    for m in models:
        v_series = (df['Realized_Return'] < -df[m]).astype(int)
        v_count = v_series.sum()
        
        lr_k, p_k = run_kupiec_test(v_count, total_days, alpha)
        lr_c, p_c = run_christoffersen_independence(v_series)
        
        # Joint Test (Conditional Coverage LR_cc = LR_pof + LR_ind, df=2)
        lr_cc = lr_k + lr_c
        p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
        
        zone = get_basel_zone(v_count, total_days)
        
        summary.append({
            "Model": m, "Violations": v_count, "Basel Zone": zone,
            "Kupiec p-val": f"{p_k:.4f}", "Indep p-val": f"{p_c:.4f}", "Joint p-val": f"{p_cc:.4f}"
        })
        
        # Genera e salva il grafico in plots/
        plt.figure(figsize=(11, 4))
        plt.plot(df.index, df['Realized_Return'], label='Realized Return', color='royalblue', alpha=0.5, lw=0.8)
        plt.plot(df.index, -df[m], label=f'{m} Threshold', color='darkorange', linestyle='--', lw=1.2)
        v_dates = df.index[v_series == 1]
        plt.scatter(v_dates, df.loc[v_dates, 'Realized_Return'], color='crimson', s=25, label='Exceptions', zorder=5)
        plt.title(f"Backtesting Framework: {m} (Zone: {zone})")
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOT_DIR, f"backtest_{m}.png"))
        plt.close()
        
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == "__main__":
    execute_backtesting_suite()

