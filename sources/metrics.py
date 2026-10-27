import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def calculate_ewma_volatility(portfolio_returns, lam=0.94):
    n = len(portfolio_returns)
    ewma_var = np.zeros(n)
    ewma_var[0] = portfolio_returns.var()
    for t in range(1, n):
        ewma_var[t] = lam * ewma_var[t-1] + (1 - lam) * (portfolio_returns.iloc[t-1] ** 2)
    return np.sqrt(ewma_var)

def generate_rolling_risk_metrics(asset_returns, portfolio_returns):
    np.random.seed(config.RANDOM_SEED)
    n = len(portfolio_returns)
    alpha = 1 - config.CONFIDENCE_LEVEL
    window = config.WINDOW_SIZE
    
    var_hist = np.full(n, np.nan)
    es_hist = np.full(n, np.nan)
    var_param_static = np.full(n, np.nan)
    var_param_ewma = np.full(n, np.nan)
    var_mc_normal = np.full(n, np.nan)
    var_mc_student = np.full(n, np.nan)
    es_mc_student = np.full(n, np.nan)  # Added matching ES for Student-t
    
    ewma_vol = calculate_ewma_volatility(portfolio_returns, lam=0.94)
    z_score = stats.norm.ppf(config.CONFIDENCE_LEVEL)
    
    print(f"🔄 Processing rolling calculation metrics ({window}-day window, CL={config.CONFIDENCE_LEVEL*100}%)...")
    
    for t in range(window, n):
        window_ret = portfolio_returns.iloc[t-window:t]
        window_asset_ret = asset_returns.iloc[t-window:t]
        
        # 1. Historical Simulation
        var_hist[t] = -np.percentile(window_ret, alpha * 100)
        tail_losses = window_ret[window_ret <= -var_hist[t]]
        es_hist[t] = -tail_losses.mean() if len(tail_losses) > 0 else var_hist[t]
        
        # 2. Parametric Static Normal
        var_param_static[t] = -(window_ret.mean() - z_score * window_ret.std())
        
        # 3. Parametric EWMA
        var_param_ewma[t] = -(window_ret.mean() - z_score * ewma_vol[t])
        
        # 4. Monte Carlo Multivariate Normal (Like-for-like Portfolio projection)
        cov_matrix = window_asset_ret.cov()
        mean_vector = window_asset_ret.mean()
        sim_asset_ret = np.random.multivariate_normal(mean_vector, cov_matrix, config.MC_SIMULATIONS)
        sim_port_ret_norm = sim_asset_ret.dot(config.WEIGHTS)
        var_mc_normal[t] = -np.percentile(sim_port_ret_norm, alpha * 100)
        
        # 5. Monte Carlo Student's t (Multivariate to remain like-for-like)
        # Using t.fit on the historical portfolio returns for robust univariate fit
        df_fit, loc_fit, scale_fit = stats.t.fit(window_ret)
        sim_port_ret_stud = stats.t.rvs(df=df_fit, loc=loc_fit, scale=scale_fit, size=config.MC_SIMULATIONS)
        var_mc_student[t] = -np.percentile(sim_port_ret_stud, alpha * 100)
        tail_losses_mc = sim_port_ret_stud[sim_port_ret_stud <= -var_mc_student[t]]
        es_mc_student[t] = -tail_losses_mc.mean() if len(tail_losses_mc) > 0 else var_mc_student[t]

    results = pd.DataFrame({
        'Realized_Return': portfolio_returns,
        'VaR_Historical': var_hist,
        'ES_Historical': es_hist,
        'VaR_Param_Static': var_param_static,
        'VaR_Param_EWMA': var_param_ewma,
        'VaR_MC_Normal': var_mc_normal,
        'VaR_MC_Student': var_mc_student,
        'ES_MC_Student': es_mc_student
    }, index=portfolio_returns.index).dropna()
    
    results.to_csv(config.RISK_RESULTS_PATH)
    print(f"✅ Risk results exported to {config.RISK_RESULTS_PATH}")
    return results

