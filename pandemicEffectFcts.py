import pandas as pd
import yfinance as yf
from typing import Dict
from dataclasses import dataclass
from typing import List, Iterable, Sequence, Tuple
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from math import floor
from scipy.stats import norm
import warnings
from scipy.stats import chi2, norm

# %% ------------------------------ LOCAL LJUNG–BOX (no statsmodels) ------------------------------

def acorr_ljungbox(x: np.ndarray,
                   lags: int | Sequence[int] = 10,
                   return_df: bool = True,
                   model_df: int = 0) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        if return_df:
            idx = [lags] if isinstance(lags, (int, np.integer)) else list(lags)
            return pd.DataFrame({"lb_stat": np.nan, "lb_pvalue": np.nan}, index=idx)
        return np.array([np.nan]), np.array([np.nan])

    if isinstance(lags, (int, np.integer)):
        lag_list = np.arange(1, int(lags) + 1, dtype=int)
    else:
        lag_list = np.array(list(lags), dtype=int)
        if lag_list.size == 0:
            if return_df:
                return pd.DataFrame({"lb_stat": [], "lb_pvalue": []})
            return np.array([]), np.array([])
        if np.any(lag_list < 1):
            raise ValueError("All lags must be >= 1")

    max_lag = int(np.max(lag_list))
    if max_lag >= n:
        raise ValueError("All lags must be less than the number of observations")

    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0 or not np.isfinite(denom):
        if return_df:
            return pd.DataFrame({"lb_stat": np.nan, "lb_pvalue": np.nan}, index=lag_list)
        return np.full(lag_list.shape, np.nan), np.full(lag_list.shape, np.nan)

    r = np.empty(max_lag + 1, dtype=float)
    r[0] = 1.0
    for k in range(1, max_lag + 1):
        r[k] = np.dot(x[k:], x[:-k]) / denom

    lb_stat = np.empty(lag_list.size, dtype=float)
    lb_pvalue = np.empty(lag_list.size, dtype=float)

    term = np.array([(r[k] ** 2) / (n - k) for k in range(1, max_lag + 1)], dtype=float)
    csum = np.cumsum(term)

    for i, h in enumerate(lag_list):
        Q = n * (n + 2.0) * csum[h - 1]
        df = max(int(h) - int(model_df), 1)
        p = 1.0 - chi2.cdf(Q, df=df)
        lb_stat[i] = Q
        lb_pvalue[i] = p

    if return_df:
        return pd.DataFrame({"lb_stat": lb_stat, "lb_pvalue": lb_pvalue}, index=lag_list)
    return lb_stat, lb_pvalue


# %% ------------------------------ LOCAL ADFULLER (no statsmodels) ------------------------------

def _ols_beta(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (beta, s2, XtX_inv) for OLS y ~ X."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    s2 = float((resid @ resid) / dof)
    return beta, s2, XtX_inv

def _adf_regression(y: np.ndarray, p: int, regression: str) -> tuple[float, int]:
    """
    ADF regression:
      Δy_t = a + b*t + γ y_{t-1} + Σ_{i=1..p} φ_i Δy_{t-i} + u_t
    regression in {"n","c","ct"}.
    Returns (t_stat for γ, nobs_used).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < (p + 5):
        return np.nan, 0

    dy = np.diff(y)                 # length n-1
    y_lag1 = y[:-1]                 # y_{t-1}, length n-1

    # build rows t = p .. (n-2) in dy index
    start = p
    end = dy.size                   # dy index goes 0..n-2, we use start..end-1
    m = end - start
    if m <= 0:
        return np.nan, 0

    Y = dy[start:end]               # dependent variable
    cols = []

    if regression in ("c", "ct"):
        cols.append(np.ones(m))
    if regression == "ct":
        t_idx = np.arange(start + 1, end + 1, dtype=float)  # 1..(n-1) aligned with dy
        cols.append(t_idx)

    cols.append(y_lag1[start:end])  # γ coefficient is on this column

    # lagged differences Δy_{t-1}..Δy_{t-p}
    for i in range(1, p + 1):
        cols.append(dy[start - i:end - i])

    X = np.column_stack(cols)
    beta, s2, XtX_inv = _ols_beta(Y, X)

    # γ is the coefficient on y_{t-1} column
    # its position depends on regression
    if regression == "n":
        gamma_idx = 0
    elif regression == "c":
        gamma_idx = 1
    else:  # "ct"
        gamma_idx = 2

    se_gamma = float(np.sqrt(s2 * XtX_inv[gamma_idx, gamma_idx]))
    t_stat = float(beta[gamma_idx] / se_gamma) if se_gamma > 0 else np.nan
    return t_stat, m

def _aic_from_ols(y: np.ndarray, X: np.ndarray) -> float:
    """Gaussian AIC for OLS."""
    beta, s2, _ = _ols_beta(y, X)
    n = y.shape[0]
    k = X.shape[1]
    # loglik up to constant: -n/2 * (log(s2)+1)
    ll = -0.5 * n * (np.log(s2) + 1.0)
    return float(2 * k - 2 * ll)

def adfuller(x: np.ndarray,
             maxlag: int | None = None,
             regression: str = "c",
             autolag: str | None = "AIC") -> tuple:
    """
    Lightweight local implementation of statsmodels.tsa.stattools.adfuller.

    Supports:
      - regression: "n", "c", "ct"
      - autolag: "AIC" (default) or None
      - returns a tuple compatible with your usage: res[0] (stat), res[1] (pvalue)

    Note: p-values here use a *normal approximation* as a simple local substitute.
    For exact MacKinnon p-values/critvals you’d need tabulated response surfaces.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return (np.nan, np.nan, None, n, None, None)

    if regression not in ("n", "c", "ct"):
        raise ValueError("regression must be one of {'n','c','ct'}")

    if maxlag is None:
        # common rule of thumb: floor(12*(n/100)^(1/4))
        maxlag = int(floor(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(int(maxlag), n - 5))

    # choose lag order
    if autolag is None:
        p_opt = maxlag
    else:
        if str(autolag).upper() != "AIC":
            raise ValueError("Only autolag='AIC' or None is supported in this local version.")
        # evaluate AIC over p=0..maxlag
        y = x
        dy = np.diff(y)
        y_lag1 = y[:-1]
        best_aic = np.inf
        p_opt = 0

        for p in range(0, maxlag + 1):
            start = p
            end = dy.size
            m = end - start
            if m <= 0:
                continue

            Y = dy[start:end]
            cols = []
            if regression in ("c", "ct"):
                cols.append(np.ones(m))
            if regression == "ct":
                t_idx = np.arange(start + 1, end + 1, dtype=float)
                cols.append(t_idx)
            cols.append(y_lag1[start:end])
            for i in range(1, p + 1):
                cols.append(dy[start - i:end - i])
            X = np.column_stack(cols)
            aic = _aic_from_ols(Y, X)
            if aic < best_aic:
                best_aic = aic
                p_opt = p

    stat, used = _adf_regression(x, p_opt, regression=regression)

    # simple p-value approximation (keeps code local; not MacKinnon exact)
    # ADF left-tail test: more negative => stronger rejection
    pval = float(norm.cdf(stat)) if np.isfinite(stat) else np.nan

    # placeholders for compatibility
    crit = None
    icbest = None
    return (stat, pval, p_opt, used, crit, icbest)


# %% ------------------------------ CONFIG ------------------------------
@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str

TICKERS: Dict[str, str] = {
    "Copper": "HG=F",
    "BCOM": "^BCOM",
}

PERIODS: List[Period] = [
    Period("2015–2019", "2015-01-01", "2019-12-31"),
    Period("2020–2024", "2020-01-01", "2024-12-31"),
]

DATE_FMT = mdates.DateFormatter("%Y-%m")
DATE_LOC = mdates.AutoDateLocator(minticks=6, maxticks=12)

# %% ------------------------------ I/O ------------------------------
def download_yahoo_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def fetch_assets(
    tickers: Dict[str, str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    return {name: download_yahoo_data(tkr, start, end, interval) for name, tkr in tickers.items()}

# %% ------------------------------ TRANSFORMS ------------------------------
def compute_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype("float64")
    out["Return"] = close.pct_change()
    out["LogReturn"] = np.log(close / close.shift(1))
    out["SqLogReturn"] = out["LogReturn"] ** 2
    out = out.dropna().reset_index().rename(columns={"index": "Date"})
    return out

def split_by_period(df_metrics: pd.DataFrame, periods: Iterable[Period]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    m = df_metrics.copy()
    m["Date"] = pd.to_datetime(m["Date"])
    for p in periods:
        mask = (m["Date"] >= p.start) & (m["Date"] <= p.end)
        out[p.name] = m.loc[mask].reset_index(drop=True)
    return out

# %% ------------------------------ PLOTTING HELPERS ------------------------------
def _format_ax_time(ax):
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(DATE_LOC)
    ax.xaxis.set_major_formatter(DATE_FMT)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

def plot_returns_grid(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]]):
    n_assets = len(metrics_by_asset)
    fig, axes = plt.subplots(n_assets, 2, figsize=(16, 3.5 * n_assets), sharey='row')
    axes = np.atleast_2d(axes)

    for r, (asset, parts) in enumerate(metrics_by_asset.items()):
        rets = [parts[p]["Return"].astype(float).to_numpy() for p in parts]
        y_min, y_max = float(np.nanmin([x.min() for x in rets])), float(np.nanmax([x.max() for x in rets]))
        for c, (pname, dfp) in enumerate(parts.items()):
            ax = axes[r, c]
            ax.plot(dfp["Date"], dfp["Return"])
            ax.set_title(f"{asset} Returns ({pname})")
            if c == 0:
                ax.set_ylabel("Returns")
            ax.set_ylim(y_min, y_max)
            _format_ax_time(ax)

    fig.suptitle("Returns Comparison Across Periods", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()

def plot_overlays(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]], periods: Iterable[Period]):
    fig, axes = plt.subplots(len(metrics_by_asset), 1, figsize=(16, 4.5 * len(metrics_by_asset)), sharex=False)
    axes = np.atleast_1d(axes)

    pnames = [p.name for p in periods]
    for ax, (asset, parts) in zip(axes, metrics_by_asset.items()):
        for pname in pnames:
            dfp = parts[pname]
            ax.plot(dfp["Date"], dfp["Return"], label=pname, alpha=0.9)
        ax.set_title(f"{asset} Returns: Period Overlay")
        ax.set_ylabel("Returns")
        ax.legend()
        _format_ax_time(ax)

    plt.tight_layout()
    plt.show()

# %% ------------------------------ SIMPLE STATS ------------------------------
def period_volatility(dfp: pd.DataFrame) -> float:
    return float(dfp["SqLogReturn"].std())

def summarize_volatilities(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = [
        {"Asset": asset, "Period": pname, "Volatility(Std[SqLogRet])": period_volatility(dfp)}
        for asset, parts in metrics_by_asset.items()
        for pname, dfp in parts.items()
    ]
    return pd.DataFrame(rows).pivot(index="Asset", columns="Period", values="Volatility(Std[SqLogRet])")

# %% ------------------------------ HAC UTILITIES ------------------------------
def _autobandwidth_newey_west(n: int) -> int:
    q = int(floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(q, 1)

def long_run_variance_newey_west(x: np.ndarray, q: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.shape[0]
    if n < 3:
        return np.nan
    if q is None:
        q = _autobandwidth_newey_west(n)
    xc = x - x.mean()
    gamma0 = np.dot(xc, xc) / n
    lrv = gamma0 + sum(
        2.0 * (1.0 - h / (q + 1.0)) * (np.dot(xc[h:], xc[:-h]) / n)
        for h in range(1, q + 1)
    )
    return float(lrv)

def hac_two_sample_mean_test(x: np.ndarray, y: np.ndarray, qx: int | None = None, qy: int | None = None,
                             alternative: str = "two-sided") -> dict:
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    y = np.asarray(y, dtype=float); y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return {"stat": np.nan, "pvalue": np.nan, "se": np.nan, "n1": n1, "n2": n2}
    qx = _autobandwidth_newey_west(n1) if qx is None else qx
    qy = _autobandwidth_newey_west(n2) if qy is None else qy
    lrv_x = long_run_variance_newey_west(x, q=qx)
    lrv_y = long_run_variance_newey_west(y, q=qy)
    se = np.sqrt(lrv_x / n1 + lrv_y / n2)
    diff = x.mean() - y.mean()
    z = diff / se if se > 0 else np.nan
    pval = (
        2.0 * (1.0 - norm.cdf(abs(z))) if alternative == "two-sided"
        else (1.0 - norm.cdf(z) if alternative == "larger" else norm.cdf(z))
    )
    return {
        "mean_x": x.mean(), "mean_y": y.mean(),
        "diff": diff, "se": se, "stat": z, "pvalue": pval,
        "lags_x": qx, "lags_y": qy, "n1": n1, "n2": n2
    }

def pretty_test(name: str, res: dict) -> dict:
    return {
        "Test": name, "n1": res.get("n1"), "n2": res.get("n2"),
        "mean_x": res.get("mean_x"), "mean_y": res.get("mean_y"),
        "diff": res.get("diff"), "HAC_SE": res.get("se"),
        "z_stat": res.get("stat"), "p_value": res.get("pvalue"),
        "lags_x": res.get("lags_x"), "lags_y": res.get("lags_y"),
    }

# %% ------------------------------ DIAGNOSTICS ------------------------------
def adf_test(x: np.ndarray) -> Tuple[float, float]:
    # H0: unit root (non-stationary). Prefer to REJECT H0.
    res = adfuller(x, autolag="AIC", regression="c")
    return res[0], res[1]

def lb_test(x: np.ndarray, lags: Iterable[int] = (5, 10, 20)) -> Dict[int, float]:
    return {L: float(acorr_ljungbox(x, lags=[L], return_df=True)["lb_pvalue"].iloc[0]) for L in lags}

def rolling_cv(x: np.ndarray, window: int = 126) -> float:
    s = pd.Series(x).dropna()
    if len(s) < window * 2:
        return np.nan
    rv = s.rolling(window).var()
    return float(rv.std() / rv.mean())

def mean_zero_and_variance_checks(r: np.ndarray, eps: float = 1e-14) -> Tuple[float, float, float, float, bool, bool]:
    r = pd.Series(r).dropna().values
    n = len(r)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, True, True
    mean_r = float(np.mean(r))
    var_r = float(np.var(r, ddof=1))
    se_mean = np.sqrt(var_r / n) if var_r > 0 else np.nan
    z_mean = mean_r / se_mean if (np.isfinite(se_mean) and se_mean > 0) else np.nan
    p_mean = 2 * (1 - norm.cdf(abs(z_mean))) if np.isfinite(z_mean) else np.nan
    zero_var_returns = bool(var_r <= eps)

    sq = r**2
    var_sq = float(np.var(sq, ddof=1))
    zero_var_sq = bool(var_sq <= eps)
    return mean_r, var_r, float(z_mean), float(p_mean), zero_var_returns, zero_var_sq

def check_assumptions(name: str, ret_series: np.ndarray, sq_series: np.ndarray) -> dict:
    x = pd.Series(sq_series).dropna().values
    r = pd.Series(ret_series).dropna().values
    n = len(x)
    out = {"series": name, "n": n}

    mean_r, var_r, z_mean, p_mean, zero_var_r, zero_var_sq = mean_zero_and_variance_checks(r)
    out.update({
        "Mean_LogRet": mean_r, "Var_LogRet": var_r,
        "z_mean0": z_mean, "p_mean0": p_mean,
        "ZeroVar_Returns": zero_var_r, "ZeroVar_SqReturns": zero_var_sq
    })

    if n >= 10 and not zero_var_sq:
        adf_stat, adf_p = adf_test(x)
        lb_p = lb_test(x, lags=(5, 10, 20)) if n >= 20 else {5: np.nan, 10: np.nan, 20: np.nan}
        lrv = long_run_variance_newey_west(x)
    else:
        adf_stat = adf_p = lrv = np.nan
        lb_p = {5: np.nan, 10: np.nan, 20: np.nan}

    out.update({
        "ADF_stat": adf_stat, "ADF_p": adf_p,
        "LB_p_lag5": lb_p[5], "LB_p_lag10": lb_p[10], "LB_p_lag20": lb_p[20],
        "LRV_NW": lrv,
        "RollVar_CV_126": rolling_cv(x, window=126) if n >= 252 else np.nan,
    })
    return out

def safe_hac(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    cond_bad = (
        len(x) < 3 or len(y) < 3 or
        np.var(x, ddof=1) <= 1e-14 or np.var(y, ddof=1) <= 1e-14
    )
    if cond_bad:
        return pretty_test(label, {
            "n1": len(x), "n2": len(y),
            "mean_x": float(np.mean(x)) if len(x) else np.nan,
            "mean_y": float(np.mean(y)) if len(y) else np.nan,
            "diff": (float(np.mean(x)) - float(np.mean(y))) if len(x) and len(y) else np.nan,
            "se": np.nan, "stat": np.nan, "pvalue": np.nan,
            "lags_x": np.nan, "lags_y": np.nan
        })
    return pretty_test(label, hac_two_sample_mean_test(x, y))
