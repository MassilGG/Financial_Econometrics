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

# Constants and configurations
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

@dataclass(frozen=True)
class Period:
    """
    Simple container for named time windows.

    Fields:
      - name: label shown in tables/plots (e.g., "2015–2019")
      - start: start date string (YYYY-MM-DD)
      - end: end date string (YYYY-MM-DD)
    """
    name: str
    start: str
    end: str



# Ljung–Box test implementation

def acorr_ljungbox(x: np.ndarray,
                   lags: int | Sequence[int] = 10,
                   return_df: bool = True,
                   model_df: int = 0) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
    """
    Ljung–Box Q-test for (joint) autocorrelation up to given lag(s).

    Inputs:
      - x: 1D array-like time series (will be demeaned internally).
      - lags: int (max lag, tests 1..lags) or sequence of specific lags.
      - return_df: if True, returns a pandas DataFrame; otherwise returns (Q, p-values) arrays.
      - model_df: degrees-of-freedom correction (e.g., number of ARMA parameters fitted).

    Outputs:
      - If return_df=True: DataFrame indexed by lag with columns:
          * lb_stat: Ljung–Box Q statistic
          * lb_pvalue: chi-square p-value
      - Else: (lb_stat_array, lb_pvalue_array)
    """
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


# Augmented Dickey–Fuller test implementation

def _ols_beta(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Ordinary Least Squares (OLS) helper for y ~ X.

    Inputs:
      - y: dependent variable (1D array, length n)
      - X: design matrix (2D array, shape n x k)

    Outputs:
      - beta: OLS coefficients (k,)
      - s2: residual variance estimate (float)
      - XtX_inv: pseudo-inverse of X'X (k x k), used for standard errors
    """
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
    Compute the ADF regression t-statistic for the unit-root coefficient.

    Model:
      Δy_t = a + b*t + γ y_{t-1} + Σ_{i=1..p} φ_i Δy_{t-i} + u_t
    where regression in {"n","c","ct"} controls deterministic terms.

    Inputs:
      - y: 1D array time series
      - p: number of lagged differences included (ADF lag order)
      - regression: "n" (none), "c" (constant), or "ct" (constant + trend)

    Outputs:
      - t_stat: t-statistic for γ (unit-root coefficient)
      - nobs_used: number of effective observations used in the regression
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < (p + 5):
        return np.nan, 0

    dy = np.diff(y)
    y_lag1 = y[:-1]

    start = p
    end = dy.size
    m = end - start
    if m <= 0:
        return np.nan, 0

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
    beta, s2, XtX_inv = _ols_beta(Y, X)

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
    """
    Compute Gaussian AIC for an OLS regression.

    Inputs:
      - y: dependent variable
      - X: design matrix

    Outputs:
      - aic: Akaike Information Criterion (float)
    """
    beta, s2, _ = _ols_beta(y, X)
    n = y.shape[0]
    k = X.shape[1]
    ll = -0.5 * n * (np.log(s2) + 1.0)  # log-likelihood up to a constant
    return float(2 * k - 2 * ll)


def adfuller(x: np.ndarray,
             maxlag: int | None = None,
             regression: str = "c",
             autolag: str | None = "AIC") -> tuple:
    """
    Lightweight Augmented Dickey–Fuller (ADF) unit root test.

    Hypotheses:
      - H0: unit root (non-stationary)
      - H1: stationary

    Inputs:
      - x: 1D array-like time series
      - maxlag: maximum lag order to consider (if None, uses a rule-of-thumb)
      - regression: deterministic terms ("n", "c", or "ct")
      - autolag: if "AIC", selects lag order minimizing AIC; if None, uses maxlag

    Outputs (tuple for compatibility):
      - stat: ADF t-statistic for the unit-root coefficient
      - pvalue: approximate p-value (normal approximation in this local version)
      - usedlag: selected lag order
      - nobs: number of observations used in the final regression
      - critvalues: placeholder (None)
      - icbest: placeholder (None)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return (np.nan, np.nan, None, n, None, None)

    if regression not in ("n", "c", "ct"):
        raise ValueError("regression must be one of {'n','c','ct'}")

    if maxlag is None:
        maxlag = int(floor(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(int(maxlag), n - 5))

    if autolag is None:
        p_opt = maxlag
    else:
        if str(autolag).upper() != "AIC":
            raise ValueError("Only autolag='AIC' or None is supported.")
        dy = np.diff(x)
        y_lag1 = x[:-1]
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
    pval = float(norm.cdf(stat)) if np.isfinite(stat) else np.nan  # left-tail approx
    return (stat, pval, p_opt, used, None, None)


# I/O

def download_yahoo_data(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance via yfinance.

    Inputs:
      - ticker: Yahoo ticker symbol (e.g., "HG=F", "^BCOM")
      - start: start date (YYYY-MM-DD)
      - end: end date (YYYY-MM-DD)
      - interval: sampling frequency (default "1d")

    Output:
      - DataFrame indexed by date, sorted in ascending order.
    """
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
    """
    Download price data for multiple assets.

    Inputs:
      - tickers: dict mapping {asset_name: yahoo_ticker}
      - start, end, interval: passed to the downloader

    Output:
      - dict mapping {asset_name: raw price DataFrame}
    """
    return {name: download_yahoo_data(tkr, start, end, interval) for name, tkr in tickers.items()}


# Data processing

def compute_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute return metrics from raw price data.

    Inputs:
      - df: Yahoo Finance DataFrame containing a 'Close' column

    Output:
      - DataFrame with columns:
          * Date: trading date
          * Return: simple return
          * LogReturn: log return
          * SqLogReturn: squared log return (volatility proxy)
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype("float64")
    out["Return"] = close.pct_change()
    out["LogReturn"] = np.log(close / close.shift(1))
    out["SqLogReturn"] = out["LogReturn"] ** 2
    out = out.dropna().reset_index().rename(columns={"index": "Date"})
    return out


def split_by_period(df_metrics: pd.DataFrame, periods: Iterable[Period]) -> Dict[str, pd.DataFrame]:
    """
    Split a metrics DataFrame into multiple named subperiods.

    Inputs:
      - df_metrics: DataFrame with a 'Date' column
      - periods: iterable of Period objects

    Output:
      - dict mapping {period_name: DataFrame restricted to that period}
    """
    out: Dict[str, pd.DataFrame] = {}
    m = df_metrics.copy()
    m["Date"] = pd.to_datetime(m["Date"])
    for p in periods:
        mask = (m["Date"] >= p.start) & (m["Date"] <= p.end)
        out[p.name] = m.loc[mask].reset_index(drop=True)
    return out


# Plot helpers

def _format_ax_time(ax):
    """
    Format a matplotlib axis with readable date ticks and a light grid.

    Input:
      - ax: matplotlib axis

    Output:
      - None (modifies axis in-place)
    """
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(DATE_LOC)
    ax.xaxis.set_major_formatter(DATE_FMT)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def plot_returns_grid(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]]):
    """
    Plot simple returns in a grid: one row per asset, one column per period.

    Inputs:
      - metrics_by_asset: dict {asset: {period: metrics_df}}

    Output:
      - None (displays matplotlib figures)
    """
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
    """
    Overlay return series across periods for each asset (one subplot per asset).

    Inputs:
      - metrics_by_asset: dict {asset: {period: metrics_df}}
      - periods: iterable of Period objects (used for consistent ordering)

    Output:
      - None (displays matplotlib figures)
    """
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


# Volatility summaries

def period_volatility(dfp: pd.DataFrame) -> float:
    """
    Compute a simple volatility summary for a period: std of squared log-returns.

    Input:
      - dfp: period-specific metrics DataFrame containing 'SqLogReturn'

    Output:
      - float: standard deviation of squared log-returns
    """
    return float(dfp["SqLogReturn"].std())


def summarize_volatilities(metrics_by_asset: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    Build a pivot table of volatility summaries by asset and period.

    Input:
      - metrics_by_asset: dict {asset: {period: metrics_df}}

    Output:
      - DataFrame: pivot with index=Asset, columns=Period, values=Volatility(Std[SqLogRet])
    """
    rows = [
        {"Asset": asset, "Period": pname, "Volatility(Std[SqLogRet])": period_volatility(dfp)}
        for asset, parts in metrics_by_asset.items()
        for pname, dfp in parts.items()
    ]
    return pd.DataFrame(rows).pivot(index="Asset", columns="Period", values="Volatility(Std[SqLogRet])")


# HAC / Newey–West robust tests

def _autobandwidth_newey_west(n: int) -> int:
    """
    Automatic bandwidth selection for Newey–West (rule-of-thumb).

    Input:
      - n: sample size

    Output:
      - q: integer bandwidth (number of lags)
    """
    q = int(floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    return max(q, 1)


def long_run_variance_newey_west(x: np.ndarray, q: int | None = None) -> float:
    """
    Estimate the long-run variance (LRV) using Newey–West with Bartlett weights.

    Inputs:
      - x: 1D array-like series
      - q: bandwidth (if None, uses automatic selection)

    Output:
      - float: Newey–West long-run variance estimate
    """
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
    """
    Two-sample test for difference in means with HAC/Newey–West standard errors.

    Hypothesis (two-sided):
      H0: E[x] - E[y] = 0

    Inputs:
      - x, y: 1D arrays for sample 1 and sample 2
      - qx, qy: Newey–West bandwidths for each sample (auto if None)
      - alternative: "two-sided", "larger" (x>y), or "smaller" (x<y)

    Output (dict):
      - mean_x, mean_y: sample means
      - diff: mean_x - mean_y
      - se: HAC standard error of diff
      - stat: z-statistic (diff / se)
      - pvalue: p-value based on N(0,1)
      - lags_x, lags_y: bandwidths used
      - n1, n2: sample sizes
    """
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
    """
    Format HAC test output into a consistent dictionary schema.

    Inputs:
      - name: label describing the test
      - res: dict returned by hac_two_sample_mean_test

    Output:
      - dict with standardized keys used to build summary tables
    """
    return {
        "Test": name, "n1": res.get("n1"), "n2": res.get("n2"),
        "mean_x": res.get("mean_x"), "mean_y": res.get("mean_y"),
        "diff": res.get("diff"), "HAC_SE": res.get("se"),
        "z_stat": res.get("stat"), "p_value": res.get("pvalue"),
        "lags_x": res.get("lags_x"), "lags_y": res.get("lags_y"),
    }


# Diagnostic checks

def adf_test(x: np.ndarray) -> Tuple[float, float]:
    """
    Convenience wrapper for the ADF test (stationarity check).

    Input:
      - x: 1D array-like series

    Output:
      - (adf_stat, adf_p): ADF statistic and p-value
    """
    res = adfuller(x, autolag="AIC", regression="c")
    return res[0], res[1]


def lb_test(x: np.ndarray, lags: Iterable[int] = (5, 10, 20)) -> Dict[int, float]:
    """
    Convenience wrapper computing Ljung–Box p-values at selected lags.

    Inputs:
      - x: 1D array-like series
      - lags: iterable of lags (default 5, 10, 20)

    Output:
      - dict {lag: p-value}
    """
    return {L: float(acorr_ljungbox(x, lags=[L], return_df=True)["lb_pvalue"].iloc[0]) for L in lags}


def rolling_cv(x: np.ndarray, window: int = 126) -> float:
    """
    Rolling variance stability metric: coefficient of variation of rolling variance.

    Inputs:
      - x: 1D array-like series
      - window: rolling window length (default 126 ~ half-year of trading days)

    Output:
      - float: std(rolling_var) / mean(rolling_var); higher => more time-variation
    """
    s = pd.Series(x).dropna()
    if len(s) < window * 2:
        return np.nan
    rv = s.rolling(window).var()
    return float(rv.std() / rv.mean())


def mean_zero_and_variance_checks(r: np.ndarray, eps: float = 1e-14) -> Tuple[float, float, float, float, bool, bool]:
    """
    Basic sanity checks on returns: mean near zero and non-degenerate variance.

    Inputs:
      - r: 1D array of returns
      - eps: small threshold to flag near-zero variance

    Outputs:
      - mean_r: sample mean
      - var_r: sample variance
      - z_mean: z-stat for mean=0
      - p_mean: p-value for mean=0
      - zero_var_returns: True if var(r) <= eps
      - zero_var_sq: True if var(r^2) <= eps
    """
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
    """
    Collect assumption diagnostics needed to justify HAC testing on volatility proxies.

    Inputs:
      - name: label for the series (asset + period)
      - ret_series: raw log-return series (for mean/variance sanity checks)
      - sq_series: squared log-return series (volatility proxy)

    Output:
      - dict with:
          * sample size, mean/variance checks
          * ADF stationarity test on squared returns
          * Ljung–Box p-values on squared returns (serial dependence)
          * Newey–West long-run variance
          * rolling variance stability proxy
          * degeneracy flags (zero variance)
    """
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
    """
    Safe wrapper around the two-sample HAC test with basic degeneracy checks.

    Inputs:
      - x, y: 1D arrays for two samples
      - label: descriptive test name (used in output table)

    Output:
      - dict formatted by pretty_test(), with NaNs if data are insufficient/degenerate
    """
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
