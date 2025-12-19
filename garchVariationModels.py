import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from typing import Dict
from pandemicEffectFcts import acorr_ljungbox
import matplotlib.pyplot as plt

# T-GARCH(1,1) & GARCH(1,1)

def _compute_sigma2_garch11(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """
    Compute the symmetric GARCH(1,1) conditional variance path via recursion.

    Inputs:
      - eps: 1D array of (demeaned) returns/innovations ε_t
      - omega: GARCH constant term ω (>0)
      - alpha: ARCH term α (>=0)
      - beta:  GARCH term β (>=0)

    Output:
      - sigma2: 1D array of conditional variances σ_t^2 with same length as eps
    """
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = np.var(eps, ddof=1)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    return sigma2


def _compute_sigma2_tgarch11(eps: np.ndarray, omega: float, alpha_m: float, alpha_p: float, beta: float) -> np.ndarray:
    """
    Compute the Threshold-GARCH(1,1) conditional variance path via recursion.

    Inputs:
      - eps: 1D array of (demeaned) returns/innovations ε_t
      - omega: constant term ω (>0)
      - alpha_m: ARCH coefficient for negative shocks α_- (>=0)
      - alpha_p: ARCH coefficient for positive shocks α_+ (>=0)
      - beta: GARCH coefficient β (>=0)

    Output:
      - sigma2: 1D array of conditional variances σ_t^2
    """
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    sigma2[0] = np.var(eps, ddof=1)
    for t in range(1, n):
        e_lag = eps[t-1]
        neg = (e_lag < 0.0)
        pos = (e_lag > 0.0)
        sigma2[t] = (
            omega
            + alpha_m * (e_lag**2) * neg
            + alpha_p * (e_lag**2) * pos
            + beta * sigma2[t-1]
        )
    return sigma2


def _loglik_gaussian(eps: np.ndarray, sigma2: np.ndarray) -> float:
    """
    Gaussian conditional log-likelihood (up to an additive constant).

    Inputs:
      - eps: 1D array of innovations ε_t
      - sigma2: 1D array of conditional variances σ_t^2 (same length)

    Output:
      - ll: scalar log-likelihood value (float)
    """
    eps_ = eps[1:]
    s2_ = sigma2[1:]
    return -0.5 * np.sum(np.log(s2_) + (eps_**2) / s2_)


def fit_garch11(eps: np.ndarray) -> dict:
    """
    Quasi-Maximum Likelihood (Gaussian QML) estimation of symmetric GARCH(1,1):

        σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2

    Input:
      - eps: 1D array of (demeaned) returns/innovations

    Output (dict):
      - success: optimizer success flag (bool)
      - message: optimizer message (str)
      - params: {"omega","alpha","beta"}
      - sigma2: fitted conditional variance path
      - ll: maximized log-likelihood
      - nobs: sample size used
      - stationary: True if α + β < 1 (weak stationarity check)
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[~np.isnan(eps)]
    n = len(eps)
    if n < 10:
        return {"success": False, "message": "Too few obs for GARCH(1,1)"}

    var_eps = np.var(eps, ddof=1)
    init = np.array([0.1 * var_eps, 0.05, 0.9])  # ω, α, β

    bounds = [(1e-8, None), (0.0, None), (0.0, None)]

    def objective(theta):
        omega, alpha, beta = theta
        if alpha + beta >= 0.999:
            return 1e10
        sigma2 = _compute_sigma2_garch11(eps, omega, alpha, beta)
        ll = _loglik_gaussian(eps, sigma2)
        return -ll

    res = minimize(objective, init, bounds=bounds, method="L-BFGS-B")
    omega, alpha, beta = res.x
    sigma2_hat = _compute_sigma2_garch11(eps, omega, alpha, beta)
    ll_max = -res.fun

    return {
        "success": res.success,
        "message": res.message,
        "params": {"omega": omega, "alpha": alpha, "beta": beta},
        "sigma2": sigma2_hat,
        "ll": ll_max,
        "nobs": n,
        "stationary": bool(alpha + beta < 1.0),
    }


def fit_tgarch11(eps: np.ndarray) -> dict:
    """
    Quasi-Maximum Likelihood (Gaussian QML) estimation of Threshold-GARCH(1,1):

        ε_t = σ_t Z_t
        σ_t^2 = ω + α_- ε_{t-1}^2 1_{ε_{t-1}<0}
                 + α_+ ε_{t-1}^2 1_{ε_{t-1}>0}
                 + β σ_{t-1}^2

    Input:
      - eps: 1D array of (demeaned) returns/innovations

    Output (dict):
      - success, message: optimizer outcome
      - params: {"omega","alpha_-","alpha_+","beta"}
      - sigma2: fitted conditional variance path
      - ll: maximized log-likelihood
      - nobs: sample size used
      - stationary: True if α_- + α_+ + β < 1 (sufficient stability check)
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[~np.isnan(eps)]
    n = len(eps)
    if n < 10:
        return {"success": False, "message": "Too few obs for T-GARCH(1,1)"}

    var_eps = np.var(eps, ddof=1)
    init = np.array([0.1 * var_eps, 0.05, 0.05, 0.9])  # ω, α_-, α_+, β

    bounds = [(1e-8, None), (0.0, None), (0.0, None), (0.0, None)]

    def objective(theta):
        omega, alpha_m, alpha_p, beta = theta
        if (alpha_m + alpha_p + beta) >= 0.999:
            return 1e10
        sigma2 = _compute_sigma2_tgarch11(eps, omega, alpha_m, alpha_p, beta)
        ll = _loglik_gaussian(eps, sigma2)
        return -ll

    res = minimize(objective, init, bounds=bounds, method="L-BFGS-B")
    omega, alpha_m, alpha_p, beta = res.x
    sigma2_hat = _compute_sigma2_tgarch11(eps, omega, alpha_m, alpha_p, beta)
    ll_max = -res.fun

    return {
        "success": res.success,
        "message": res.message,
        "params": {"omega": omega, "alpha_-": alpha_m, "alpha_+": alpha_p, "beta": beta},
        "sigma2": sigma2_hat,
        "ll": ll_max,
        "nobs": n,
        "stationary": bool(alpha_m + alpha_p + beta < 1.0),
    }


# LR Test and Diagnostics

def lr_test_tgarch_vs_garch(fit_sym: dict, fit_thr: dict) -> dict:
    """
    Likelihood-Ratio (LR) test for asymmetry in T-GARCH vs symmetric GARCH.

    Hypotheses:
      - H0: α_- = α_+  (no threshold effect → symmetric GARCH)
      - H1: α_- ≠ α_+  (threshold effect)

    Inputs:
      - fit_sym: dict from fit_garch11()
      - fit_thr: dict from fit_tgarch11()

    Output (dict):
      - LR: LR statistic (float)
      - pvalue: chi-square(1) p-value (float)
      - df: degrees of freedom (1)
    """
    if (not fit_sym.get("success")) or (not fit_thr.get("success")):
        return {"LR": np.nan, "pvalue": np.nan, "df": 1}

    ll0 = fit_sym["ll"]
    ll1 = fit_thr["ll"]
    lr_stat = 2.0 * (ll1 - ll0)
    pval = 1.0 - chi2.cdf(lr_stat, df=1)
    return {"LR": float(lr_stat), "pvalue": float(pval), "df": 1}


def tgarch_residual_diagnostics(eps: np.ndarray, sigma2: np.ndarray) -> dict:
    """
    Residual diagnostics for a fitted (T-)GARCH model using standardized residuals z_t.

    Checks:
      - mean(z_t) ≈ 0 via z-test
      - Ljung–Box p-values on z_t (remaining autocorrelation)
      - Ljung–Box p-values on z_t^2 (remaining ARCH effects)

    Inputs:
      - eps: 1D array of innovations ε_t
      - sigma2: 1D array of fitted conditional variances σ_t^2

    Output (dict):
      - mean_z: mean of standardized residuals
      - z_stat_mean0: z-statistic for H0: mean_z = 0
      - p_mean0: p-value for mean test
      - LB_p_resid_lag{5,10,20}: Ljung–Box p-values on z_t
      - LB_p_sqres_lag{5,10,20}: Ljung–Box p-values on z_t^2
    """
    eps = np.asarray(eps, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    z = eps / np.sqrt(sigma2)
    z = z[1:]
    z2 = z**2

    mean_z = float(np.mean(z))
    var_z = float(np.var(z, ddof=1))
    se_mean = np.sqrt(var_z / len(z))
    z_stat = mean_z / se_mean if se_mean > 0 else np.nan
    p_mean0 = 2 * (1 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan

    lb_resid = acorr_ljungbox(z, lags=[5, 10, 20], return_df=True)
    lb_sq = acorr_ljungbox(z2, lags=[5, 10, 20], return_df=True)

    out = {"mean_z": mean_z, "z_stat_mean0": float(z_stat), "p_mean0": float(p_mean0)}
    for L in (5, 10, 20):
        out[f"LB_p_resid_lag{L}"] = float(lb_resid.loc[L, "lb_pvalue"])
        out[f"LB_p_sqres_lag{L}"] = float(lb_sq.loc[L, "lb_pvalue"])
    return out


def get_logreturns(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]],
                   asset: str,
                   period: str | None = None) -> np.ndarray:
    """
    Extract and de-mean log-returns ε_t for a given asset and (optionally) a specific period.

    Inputs:
      - metrics_by_asset: dict {asset: {period: metrics_df}} with column 'LogReturn'
      - asset: asset name key
      - period: period label; if None, concatenates all periods for that asset

    Output:
      - eps: 1D numpy array of demeaned log-returns (innovations proxy)
    """
    if period is None:
        dfs = [dfp[["LogReturn"]] for dfp in metrics_by_asset[asset].values()]
        ser = pd.concat(dfs, axis=0)["LogReturn"]
    else:
        ser = metrics_by_asset[asset][period]["LogReturn"]
    eps = ser.dropna().to_numpy(dtype=float)
    return eps - eps.mean()


# EGARCH(1,1)

_EABSZ = np.sqrt(2.0 / np.pi)


def _compute_sigma2_egarch11(eps: np.ndarray,
                             c: float,
                             alpha: float,
                             gamma: float,
                             lam: float) -> np.ndarray:
    """
    Compute EGARCH(1,1) conditional variance path via log-variance recursion:

        ε_t = σ_t Z_t
        log σ_t^2 = c + α g(Z_{t-1}) + γ log σ_{t-1}^2
        g(Z) = Z + λ (|Z| - E|Z|)

    Inputs:
      - eps: 1D array of innovations ε_t
      - c, alpha, gamma, lam: EGARCH parameters (λ controls leverage/asymmetry)

    Output:
      - sigma2: 1D array of conditional variances σ_t^2
    """
    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    logs2 = np.empty(n, dtype=float)

    logs2[0] = np.log(np.var(eps, ddof=1))

    for t in range(1, n):
        sigma2_prev = np.exp(logs2[t-1])
        z_prev = eps[t-1] / np.sqrt(sigma2_prev)
        z_prev = np.clip(z_prev, -10.0, 10.0)

        g_z = z_prev + lam * (np.abs(z_prev) - _EABSZ)
        logs2[t] = c + alpha * g_z + gamma * logs2[t-1]
        logs2[t] = np.clip(logs2[t], -20.0, 20.0)

    return np.exp(logs2)


def _egarch_unconstrained_to_params(theta: np.ndarray,
                                    symmetric: bool) -> tuple[float, float, float, float]:
    """
    Map unconstrained optimizer variables to EGARCH parameters (c, alpha, gamma, lambda).

    Inputs:
      - theta: unconstrained parameter vector [c_t, alpha_t, gamma_t, lam_t]
      - symmetric: if True, enforces lambda = 0

    Output:
      - (c, alpha, gamma, lam): transformed parameters
        with gamma = 0.98*tanh(gamma_t) to keep |gamma| < 0.98 for numerical stability.
    """
    c_t, alpha_t, gamma_t, lam_t = theta
    gamma = 0.98 * np.tanh(gamma_t)
    alpha = alpha_t
    lam = 0.0 if symmetric else lam_t
    c = c_t
    return c, alpha, gamma, lam


def _egarch_objective(theta: np.ndarray,
                      eps: np.ndarray,
                      symmetric: bool) -> float:
    """
    Negative Gaussian QML objective for EGARCH(1,1) (to be minimized).

    Inputs:
      - theta: unconstrained parameters
      - eps: 1D array innovations
      - symmetric: if True, fixes lambda=0

    Output:
      - negative log-likelihood (float); large penalty if sigma2 is invalid
    """
    c, alpha, gamma, lam = _egarch_unconstrained_to_params(theta, symmetric)
    sigma2 = _compute_sigma2_egarch11(eps, c, alpha, gamma, lam)
    if (not np.all(np.isfinite(sigma2))) or np.any(sigma2 <= 0):
        return 1e12
    ll = _loglik_gaussian(eps, sigma2)
    return -ll


def fit_egarch11(eps: np.ndarray, symmetric: bool) -> dict:
    """
    Gaussian QML estimation of EGARCH(1,1).

    Inputs:
      - eps: 1D array of (demeaned) innovations ε_t
      - symmetric:
          * True  -> lambda fixed at 0 (no leverage)
          * False -> lambda estimated (leverage allowed)

    Output (dict):
      - success, message: optimizer outcome
      - params: {"c","alpha","gamma","lambda"}
      - sigma2: fitted conditional variance path
      - ll: maximized log-likelihood
      - nobs: sample size used
      - stationary: True if |gamma| < 1 (persistence check)
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[~np.isnan(eps)]
    n = len(eps)
    if n < 10:
        return {"success": False, "message": "Too few obs for EGARCH(1,1)"}

    var_eps = np.var(eps, ddof=1)
    if var_eps <= 0 or not np.isfinite(var_eps):
        return {"success": False, "message": "Non-positive variance in data"}

    logv = np.log(var_eps)

    gamma0 = 0.95
    c0 = (1.0 - gamma0) * logv
    alpha0 = 0.1
    lam0 = -0.1

    gamma0_tilde = np.arctanh(gamma0 / 0.98)

    init = np.array([c0, alpha0, gamma0_tilde, 0.0 if symmetric else lam0])

    def obj(th):
        return _egarch_objective(th, eps, symmetric=symmetric)

    res = minimize(obj, init, method="L-BFGS-B")

    if (not res.success) or (not np.isfinite(res.fun)):
        res_nm = minimize(obj, init, method="Nelder-Mead",
                          options={"maxiter": 5000, "maxfev": 8000})
        if res_nm.success and np.isfinite(res_nm.fun) and (res_nm.fun < res.fun or (not res.success)):
            res = res_nm

    if (not res.success) or (not np.isfinite(res.fun)):
        return {"success": False, "message": f"EGARCH optimizer failed: {res.message}"}

    c_hat, alpha_hat, gamma_hat, lam_hat = _egarch_unconstrained_to_params(res.x, symmetric)
    sigma2_hat = _compute_sigma2_egarch11(eps, c_hat, alpha_hat, gamma_hat, lam_hat)
    ll_max = -_egarch_objective(res.x, eps, symmetric=symmetric)

    return {
        "success": True,
        "message": res.message,
        "params": {"c": c_hat, "alpha": alpha_hat, "gamma": gamma_hat, "lambda": 0.0 if symmetric else lam_hat},
        "sigma2": sigma2_hat,
        "ll": ll_max,
        "nobs": n,
        "stationary": bool(abs(gamma_hat) < 1.0),
    }


def fit_egarch11_symmetric(eps: np.ndarray) -> dict:
    """
    Fit symmetric EGARCH(1,1) with lambda fixed to 0 (no leverage).

    Input:
      - eps: 1D array innovations

    Output:
      - dict returned by fit_egarch11()
    """
    return fit_egarch11(eps, symmetric=True)


def fit_egarch11_asym(eps: np.ndarray) -> dict:
    """
    Fit asymmetric EGARCH(1,1) with lambda estimated (leverage allowed).

    Input:
      - eps: 1D array innovations

    Output:
      - dict returned by fit_egarch11()
    """
    return fit_egarch11(eps, symmetric=False)


def lr_test_egarch_sym_vs_asym(fit_sym: dict, fit_asym: dict) -> dict:
    """
    Likelihood-Ratio (LR) test for leverage in EGARCH.

    Hypotheses:
      - H0: lambda = 0 (symmetric EGARCH)
      - H1: lambda ≠ 0 (asymmetric EGARCH)

    Inputs:
      - fit_sym: dict from fit_egarch11_symmetric()
      - fit_asym: dict from fit_egarch11_asym()

    Output (dict):
      - LR: LR statistic (float)
      - pvalue: chi-square(1) p-value (float)
      - df: degrees of freedom (1)
    """
    if (not fit_sym.get("success")) or (not fit_asym.get("success")):
        return {"LR": np.nan, "pvalue": np.nan, "df": 1}

    ll0 = fit_sym["ll"]
    ll1 = fit_asym["ll"]
    lr_stat = 2.0 * (ll1 - ll0)
    pval = 1.0 - chi2.cdf(lr_stat, df=1)
    return {"LR": float(lr_stat), "pvalue": float(pval), "df": 1}


def build_vol_series(metrics_by_asset, asset: str, period: str,
                     sigma2: np.ndarray,
                     smooth_window: int = 21,
                     annualize: int = 252) -> pd.DataFrame:
    """
    Build a time-aligned DataFrame with realized-variance proxies and model-implied volatility.

    Inputs:
      - metrics_by_asset: dict {asset: {period: metrics_df}} containing 'Date' and 'LogReturn'
      - asset: asset key
      - period: period key
      - sigma2: model conditional variance array σ_t^2 (aligned or will be truncated)
      - smooth_window: rolling window for smoothing rv = eps^2 (default 21)
      - annualize: annualization factor (default 252)

    Output:
      - DataFrame with columns:
          * Date: timestamps
          * eps: demeaned innovations
          * rv: eps^2 (realized variance proxy)
          * rv_smooth: rolling mean of rv
          * sigma2_model: σ_t^2 from the model
          * vol_model: sqrt(sigma2_model) annualized
          * vol_rv: sqrt(rv) annualized
          * vol_rv_smooth: sqrt(rv_smooth) annualized
    """
    dfp = metrics_by_asset[asset][period].copy()
    dates = pd.to_datetime(dfp["Date"])
    eps = dfp["LogReturn"].to_numpy(dtype=float)
    eps = eps - np.nanmean(eps)

    sigma2 = np.asarray(sigma2, float)

    m = min(len(eps), len(sigma2), len(dates))
    eps = eps[:m]
    sigma2 = sigma2[:m]
    dates = dates.iloc[:m]

    rv = eps**2
    rv_smooth = pd.Series(rv).rolling(smooth_window, min_periods=max(5, smooth_window//3)).mean().to_numpy()

    vol_model = np.sqrt(np.maximum(sigma2, 1e-18)) * np.sqrt(annualize)
    vol_rv = np.sqrt(np.maximum(rv, 1e-18)) * np.sqrt(annualize)
    vol_rv_smooth = np.sqrt(np.maximum(rv_smooth, 1e-18)) * np.sqrt(annualize)

    return pd.DataFrame({
        "Date": dates.values,
        "eps": eps,
        "rv": rv,
        "rv_smooth": rv_smooth,
        "sigma2_model": sigma2,
        "vol_model": vol_model,
        "vol_rv": vol_rv,
        "vol_rv_smooth": vol_rv_smooth,
    })


def vol_forecast_metrics_from_sigma2(eps: np.ndarray, sigma2_model: np.ndarray, eps_floor: float = 1e-12) -> dict:
    """
    Evaluate variance forecasts against realized variance using simple loss metrics.

    Convention used here:
      - realized rv_t = eps_t^2
      - forecast f_t = sigma2_model[t] (same index)
    (If you want strict 1-step-ahead, you can compare rv[t] to f[t-1].)

    Inputs:
      - eps: 1D array innovations ε_t
      - sigma2_model: 1D array model variances f_t = σ_t^2
      - eps_floor: small lower bound to avoid log/division issues

    Output (dict):
      - n: number of valid observations
      - MSE: mean squared error of (rv - f)
      - QLIKE: QLIKE loss mean(rv/f + log(f)) (lower is better)
      - corr: correlation between rv and f (linear association)
    """
    eps = np.asarray(eps, float)
    sigma2_model = np.asarray(sigma2_model, float)

    m = np.isfinite(eps) & np.isfinite(sigma2_model)
    eps = eps[m]
    f = sigma2_model[m]
    rv = eps**2

    if len(rv) < 20:
        return {"n": len(rv), "MSE": np.nan, "QLIKE": np.nan, "corr": np.nan}

    f_safe = np.maximum(f, eps_floor)
    mse = float(np.mean((rv - f_safe) ** 2))
    qlike = float(np.mean(rv / f_safe + np.log(f_safe)))
    corr = float(np.corrcoef(rv, f_safe)[0, 1]) if len(rv) > 2 else np.nan
    return {"n": int(len(rv)), "MSE": mse, "QLIKE": qlike, "corr": corr}


def fit_and_evaluate_vol_models(metrics_by_asset,
                                asset: str,
                                period: str,
                                smooth_window: int = 21,
                                annualize: int = 252,
                                plot: bool = True) -> pd.DataFrame:
    """
    Fit multiple volatility models (GARCH, T-GARCH, EGARCH sym/asym) on a given asset-period,
    compute forecast performance metrics, and optionally plot model-implied volatility paths.

    Inputs:
      - metrics_by_asset: dict {asset: {period: metrics_df}}
      - asset: asset key
      - period: period key
      - smooth_window: rolling window for the realized-volatility proxy (default 21)
      - annualize: annualization factor (default 252)
      - plot: if True, plot model volatilities vs smoothed rv proxy

    Output:
      - summary: DataFrame with one row per model and columns:
          * Model, n, MSE, QLIKE, corr
    """
    eps = get_logreturns(metrics_by_asset, asset, period)

    results = []

    # GARCH(1,1)
    fit_g = fit_garch11(eps)
    if fit_g.get("success"):
        p = fit_g["params"]
        sigma2_g = _compute_sigma2_garch11(eps, p["omega"], p["alpha"], p["beta"])
        ser_g = build_vol_series(metrics_by_asset, asset, period, sigma2_g, smooth_window, annualize)
        met_g = vol_forecast_metrics_from_sigma2(ser_g["eps"].to_numpy(), ser_g["sigma2_model"].to_numpy())
        results.append({"Model": "GARCH(1,1)", **met_g})
    else:
        ser_g = None
        results.append({"Model": "GARCH(1,1)", "n": 0, "MSE": np.nan, "QLIKE": np.nan, "corr": np.nan})

    # T-GARCH(1,1)
    fit_t = fit_tgarch11(eps)
    if fit_t.get("success"):
        p = fit_t["params"]
        sigma2_t = _compute_sigma2_tgarch11(eps, p["omega"], p["alpha_-"], p["alpha_+"], p["beta"])
        ser_t = build_vol_series(metrics_by_asset, asset, period, sigma2_t, smooth_window, annualize)
        met_t = vol_forecast_metrics_from_sigma2(ser_t["eps"].to_numpy(), ser_t["sigma2_model"].to_numpy())
        results.append({"Model": "T-GARCH(1,1)", **met_t})
    else:
        ser_t = None
        results.append({"Model": "T-GARCH(1,1)", "n": 0, "MSE": np.nan, "QLIKE": np.nan, "corr": np.nan})

    # EGARCH symmetric
    fit_es = fit_egarch11_symmetric(eps)
    if fit_es.get("success"):
        p = fit_es["params"]
        sigma2_es = _compute_sigma2_egarch11(eps, p["c"], p["alpha"], p["gamma"], 0.0)
        ser_es = build_vol_series(metrics_by_asset, asset, period, sigma2_es, smooth_window, annualize)
        met_es = vol_forecast_metrics_from_sigma2(ser_es["eps"].to_numpy(), ser_es["sigma2_model"].to_numpy())
        results.append({"Model": "EGARCH(1,1) sym", **met_es})
    else:
        ser_es = None
        results.append({"Model": "EGARCH(1,1) sym", "n": 0, "MSE": np.nan, "QLIKE": np.nan, "corr": np.nan})

    # EGARCH asymmetric
    fit_ea = fit_egarch11_asym(eps)
    if fit_ea.get("success"):
        p = fit_ea["params"]
        sigma2_ea = _compute_sigma2_egarch11(eps, p["c"], p["alpha"], p["gamma"], p["lambda"])
        ser_ea = build_vol_series(metrics_by_asset, asset, period, sigma2_ea, smooth_window, annualize)
        met_ea = vol_forecast_metrics_from_sigma2(ser_ea["eps"].to_numpy(), ser_ea["sigma2_model"].to_numpy())
        results.append({"Model": "EGARCH(1,1) asym", **met_ea})
    else:
        ser_ea = None
        results.append({"Model": "EGARCH(1,1) asym", "n": 0, "MSE": np.nan, "QLIKE": np.nan, "corr": np.nan})

    summary = pd.DataFrame(results)

    if plot:
        base = next((s for s in (ser_g, ser_t, ser_es, ser_ea) if s is not None), None)
        if base is not None:
            plt.figure(figsize=(16, 5))
            plt.plot(base["Date"], base["vol_rv_smooth"],
                     label=f"Sampled vol proxy (sqrt rolling mean eps^2, w={smooth_window})")

            if ser_g is not None:  plt.plot(ser_g["Date"], ser_g["vol_model"], label="GARCH(1,1) vol")
            if ser_t is not None:  plt.plot(ser_t["Date"], ser_t["vol_model"], label="T-GARCH(1,1) vol")
            if ser_es is not None: plt.plot(ser_es["Date"], ser_es["vol_model"], label="EGARCH sym vol")
            if ser_ea is not None: plt.plot(ser_ea["Date"], ser_ea["vol_model"], label="EGARCH asym vol")

            plt.title(f"{asset} — {period}: Model-implied volatility vs sampled volatility proxy (annualized)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    return summary
