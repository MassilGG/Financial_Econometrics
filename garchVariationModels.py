import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import Dict

# %% ------------------------------ T-GARCH(1,1) & GARCH(1,1) ------------------------------

def _compute_sigma2_garch11(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Recursion for symmetric GARCH(1,1) conditional variance."""
    n = len(eps)
    sigma2 = np.empty(n, dtype=float)
    # simple and robust initial value: sample variance
    sigma2[0] = np.var(eps, ddof=1)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    return sigma2

def _compute_sigma2_tgarch11(eps: np.ndarray, omega: float, alpha_m: float, alpha_p: float, beta: float) -> np.ndarray:
    """Recursion for Threshold-GARCH(1,1) conditional variance."""
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
    """Gaussian conditional log-likelihood (up to the constant term)."""
    # discard first observation because its variance is 'made up'
    eps_ = eps[1:]
    s2_ = sigma2[1:]
    return -0.5 * np.sum(np.log(s2_) + (eps_**2) / s2_)

def fit_garch11(eps: np.ndarray) -> dict:
    """
    QML estimation of symmetric GARCH(1,1):
        σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
    """
    eps = np.asarray(eps, dtype=float)
    eps = eps[~np.isnan(eps)]
    n = len(eps)
    if n < 10:
        return {"success": False, "message": "Too few obs for GARCH(1,1)"}

    var_eps = np.var(eps, ddof=1)
    init = np.array([0.1 * var_eps, 0.05, 0.9])  # ω, α, β

    bounds = [(1e-8, None), (0.0, None), (0.0, None)] # imposing positivity for alpha, alpha_-, alpha_+, beta ≥ 0

    def objective(theta):
        omega, alpha, beta = theta
        # enforce weak stationarity α+β < 1 approximately
        if alpha + beta >= 0.999: # so if the sum is >= 0.999, return a large penalty
            return 1e10
        sigma2 = _compute_sigma2_garch11(eps, omega, alpha, beta)
        ll = _loglik_gaussian(eps, sigma2)
        return -ll  # minimize negative log-likelihood

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
    QML estimation of Threshold-GARCH(1,1):
        ε_t = σ_t Z_t
        σ_t^2 = ω + α_- ε_{t-1}^2 1_{ε_{t-1}<0} + α_+ ε_{t-1}^2 1_{ε_{t-1}>0} + β σ_{t-1}^2
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
        # sufficient stationarity restriction: α_- + α_+ + β < 1
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

# %% ------------------------------ LR TEST & DIAGNOSTICS ------------------------------
def lr_test_tgarch_vs_garch(fit_sym: dict, fit_thr: dict) -> dict:
    """
    Likelihood-ratio test of H0: α_- = α_+ (symmetric GARCH)
    vs H1: general T-GARCH(1,1).
    df = 1 because T-GARCH has one extra free parameter.
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
    Basic assumption checks for the fitted T-GARCH model:
      - mean of standardized residuals ~ 0
      - Ljung–Box on z_t and z_t^2
    """
    eps = np.asarray(eps, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    z = eps / np.sqrt(sigma2)
    z = z[1:]  # drop first obs where variance is arbitrary
    z2 = z**2

    mean_z = float(np.mean(z))
    var_z = float(np.var(z, ddof=1))
    se_mean = np.sqrt(var_z / len(z))
    z_stat = mean_z / se_mean if se_mean > 0 else np.nan
    p_mean0 = 2 * (1 - norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan

    lb_resid = acorr_ljungbox(z, lags=[5, 10, 20], return_df=True)
    lb_sq = acorr_ljungbox(z2, lags=[5, 10, 20], return_df=True)

    out = {
        "mean_z": mean_z,
        "z_stat_mean0": float(z_stat),
        "p_mean0": float(p_mean0),
    }
    for L in (5, 10, 20):
        out[f"LB_p_resid_lag{L}"] = float(lb_resid.loc[L, "lb_pvalue"])
        out[f"LB_p_sqres_lag{L}"] = float(lb_sq.loc[L, "lb_pvalue"])
    return out


def get_logreturns(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]],
                   asset: str,
                   period: str | None = None) -> np.ndarray:
    """
    Extract log-returns for a given asset.
    If period is None, concatenate all periods for that asset.
    """
    if period is None:
        dfs = [dfp[["LogReturn"]] for dfp in metrics_by_asset[asset].values()]
        ser = pd.concat(dfs, axis=0)["LogReturn"]
    else:
        ser = metrics_by_asset[asset][period]["LogReturn"]
    eps = ser.dropna().to_numpy(dtype=float)
    # de-mean as in the slides: E[ε_t | F_{t-1}] = 0
    return eps - eps.mean()

# %% ------------------------------ EGARCH(1,1) ------------------------------

# E[|Z|] for Z ~ N(0,1)
_EABSZ = np.sqrt(2.0 / np.pi)


def _compute_sigma2_egarch11(eps: np.ndarray,
                             c: float,
                             alpha: float,
                             gamma: float,
                             lam: float) -> np.ndarray:
    """
    EGARCH(1,1) recursion:
        ε_t = σ_t Z_t
        log σ_t^2 = c + α g(Z_{t-1}) + γ log σ_{t-1}^2
        g(Z) = Z + λ (|Z| - E|Z|)
    """
    eps = np.asarray(eps, dtype=float)
    n = len(eps)
    logs2 = np.empty(n, dtype=float)

    # init: log of sample variance
    logs2[0] = np.log(np.var(eps, ddof=1))

    for t in range(1, n):
        sigma2_prev = np.exp(logs2[t-1])
        z_prev = eps[t-1] / np.sqrt(sigma2_prev)

        # numerical safety
        z_prev = np.clip(z_prev, -10.0, 10.0)

        g_z = z_prev + lam * (np.abs(z_prev) - _EABSZ)

        logs2[t] = c + alpha * g_z + gamma * logs2[t-1]
        # keep log-variance in a reasonable numerical range
        logs2[t] = np.clip(logs2[t], -20.0, 20.0)

    sigma2 = np.exp(logs2)
    return sigma2


def _egarch_unconstrained_to_params(theta: np.ndarray,
                                    symmetric: bool) -> tuple[float, float, float, float]:
    """
    Map unconstrained parameters to (c, alpha, gamma, lambda).
    gamma = 0.98 * tanh(gamma_tilde)  ensures |gamma| < 0.98.
    """
    c_t, alpha_t, gamma_t, lam_t = theta
    gamma = 0.98 * np.tanh(gamma_t)  # stationarity-ish
    alpha = alpha_t                  # can be any real
    lam = 0.0 if symmetric else lam_t
    c = c_t
    return c, alpha, gamma, lam


def _egarch_objective(theta: np.ndarray,
                      eps: np.ndarray,
                      symmetric: bool) -> float:
    """Negative Gaussian QML log-likelihood for EGARCH."""
    c, alpha, gamma, lam = _egarch_unconstrained_to_params(theta, symmetric)
    sigma2 = _compute_sigma2_egarch11(eps, c, alpha, gamma, lam)

    # penalize any numerical nonsense
    if (not np.all(np.isfinite(sigma2))) or np.any(sigma2 <= 0):
        return 1e12

    ll = _loglik_gaussian(eps, sigma2)
    return -ll


def fit_egarch11(eps: np.ndarray, symmetric: bool) -> dict:
    """
    QML estimation of EGARCH(1,1).

    symmetric = True  →  λ = 0  (no leverage, H0)
    symmetric = False →  λ free (leverage allowed, H1)
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

    # Nelson-style: choose gamma0, then c0 so that E(log σ^2) ≈ logv
    gamma0 = 0.95
    c0 = (1.0 - gamma0) * logv
    alpha0 = 0.1
    lam0 = -0.1  # typical leverage sign for equities; for commodities could be 0.0

    # inverse of gamma = 0.98 * tanh(gamma_tilde)
    gamma0_tilde = np.arctanh(gamma0 / 0.98)

    if symmetric:
        init = np.array([c0, alpha0, gamma0_tilde, 0.0])
    else:
        init = np.array([c0, alpha0, gamma0_tilde, lam0])

    def obj(th):
        return _egarch_objective(th, eps, symmetric=symmetric)

    # 1st attempt: L-BFGS-B
    res = minimize(obj, init, method="L-BFGS-B")

    # Fallback: Nelder–Mead if necessary
    if (not res.success) or (not np.isfinite(res.fun)):
        res_nm = minimize(obj, init, method="Nelder-Mead",
                          options={"maxiter": 5000, "maxfev": 8000})
        if res_nm.success and np.isfinite(res_nm.fun) and (res_nm.fun < res.fun or (not res.success)):
            res = res_nm

    if (not res.success) or (not np.isfinite(res.fun)):
        return {
            "success": False,
            "message": f"EGARCH optimizer failed: {res.message}",
        }

    c_hat, alpha_hat, gamma_hat, lam_hat = _egarch_unconstrained_to_params(res.x, symmetric)
    sigma2_hat = _compute_sigma2_egarch11(eps, c_hat, alpha_hat, gamma_hat, lam_hat)
    ll_max = -_egarch_objective(res.x, eps, symmetric=symmetric)

    return {
        "success": True,
        "message": res.message,
        "params": {
            "c": c_hat,
            "alpha": alpha_hat,
            "gamma": gamma_hat,
            "lambda": 0.0 if symmetric else lam_hat,
        },
        "sigma2": sigma2_hat,
        "ll": ll_max,
        "nobs": n,
        "stationary": bool(abs(gamma_hat) < 1.0),
    }


def fit_egarch11_symmetric(eps: np.ndarray) -> dict:
    """Wrapper: EGARCH(1,1) with λ = 0."""
    return fit_egarch11(eps, symmetric=True)


def fit_egarch11_asym(eps: np.ndarray) -> dict:
    """Wrapper: EGARCH(1,1) with free λ (leverage allowed)."""
    return fit_egarch11(eps, symmetric=False)


def lr_test_egarch_sym_vs_asym(fit_sym: dict, fit_asym: dict) -> dict:
    """
    LR test of H0: λ = 0 (symmetric EGARCH)
          vs H1: λ ≠ 0 (asymmetric EGARCH).
    """
    if (not fit_sym.get("success")) or (not fit_asym.get("success")):
        return {"LR": np.nan, "pvalue": np.nan, "df": 1}

    ll0 = fit_sym["ll"]
    ll1 = fit_asym["ll"]
    lr_stat = 2.0 * (ll1 - ll0)
    pval = 1.0 - chi2.cdf(lr_stat, df=1)
    return {"LR": float(lr_stat), "pvalue": float(pval), "df": 1}
