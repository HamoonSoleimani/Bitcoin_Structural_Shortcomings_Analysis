#!/usr/bin/env python3
# ==============================================================================
#   REPRODUCIBLE ANALYSIS SCRIPT
# ------------------------------------------------------------------------------
#   Title:   Bitcoin's Structural Position as Money: A Contested Synthesis
#   Author:  Hamoon Soleimani
#
# REQUIRED DEPENDENCIES
# ------------------------------------------------------------------------------
#   pip install yfinance arch
# ==============================================================================

import os
import json
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False


# ==============================================================================
# 1. GLOBAL CONFIGURATION
# ==============================================================================

FINAL_ANALYSIS_DATE = '2026-09-05'
START_DATE_DRAWDOWN = '2015-01-01'
FULL_START_DATE = START_DATE_DRAWDOWN

TICKERS = {
    'Bitcoin': 'BTC-USD',
    'US Dollar': 'UUP',
    'Gold': 'GC=F',
    'S&P 500': '^GSPC',
    'VIX': '^VIX',
}
ASSETS_FOR_VOL_COMP = {"AAPL": "Apple", "BTC-USD": "Bitcoin", "GC=F": "Gold"}
CACHE_FILENAME = "research_data_static_v5.csv"

TRADING_DAYS = {
    'Bitcoin': 365, 'Apple': 252, 'Gold': 252, 'S&P 500': 252, 'US Dollar': 252,
    'VIX': 252,
}

OUTPUT_DIR = "paper_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CB = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#7F7F7F",
}

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, ValueError):
    plt.style.use('default')

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "grid.color": "#cccccc", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "font.family": "serif", "axes.labelsize": 12, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "savefig.dpi": 300,
})


# ==============================================================================
# 2. OUTPUT HELPERS
# ==============================================================================

def save_fig(fig, basename):
    note = getattr(fig, "_pending_footnote", None)
    if note:
        import textwrap
        fig_w_in = fig.get_size_inches()[0]
        chars_per_line = max(60, int(fig_w_in * 13))
        wrapped = "\n".join(textwrap.wrap(note, width=chars_per_line))
        n_lines = wrapped.count("\n") + 1
        current_bottom = fig.subplotpars.bottom
        extra_needed = 0.018 + 0.026 * n_lines
        fig.subplots_adjust(bottom=min(0.45, current_bottom + extra_needed))
        fig.text(0.5, 0.004, wrapped, ha='center', va='bottom', fontsize=7.5,
                  style='italic', color='#444444')
    pdf_path = os.path.join(OUTPUT_DIR, f"{basename}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  -> saved {pdf_path} (vector) and {png_path} (preview)")


def export_table(df, basename, caption="", label=""):
    csv_path = os.path.join(OUTPUT_DIR, f"{basename}.csv")
    tex_path = os.path.join(OUTPUT_DIR, f"{basename}.tex")
    df.to_csv(csv_path, index=False)
    try:
        tex = df.to_latex(index=False, float_format="%.4f", caption=caption, label=label, escape=True)
    except TypeError:
        tex = df.to_latex(index=False, float_format="%.4f", escape=True)
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"  -> saved {csv_path} and {tex_path}")


def footnote(fig, text):
    fig._pending_footnote = text


# ==============================================================================
# 3. MARKET DATA LOADING
# ==============================================================================

def get_market_data(start_date, end_date, cache_filename):
    if not HAS_YFINANCE:
        print("  [ERROR] `yfinance` is not installed. Run: pip install yfinance")
        return pd.DataFrame()

    all_tickers_needed = sorted(set(list(TICKERS.values()) + list(ASSETS_FOR_VOL_COMP.keys())))

    if os.path.exists(cache_filename):
        print(f"Loading data from static cache: {cache_filename}...")
        try:
            data = pd.read_csv(cache_filename, index_col='Date', parse_dates=True)
            if not data.empty and all(t in data.columns for t in all_tickers_needed):
                return data.dropna()
            print("Cache is empty or incomplete. Refetching.")
        except Exception as e:
            print(f"Error loading cache ({e}). Refetching.")

    print("Fetching data from yfinance...")
    try:
        data = yf.download(all_tickers_needed, start=start_date, end=end_date)['Close'].dropna()
        if data.empty:
            raise ValueError("yfinance returned an empty DataFrame.")
        data.to_csv(cache_filename)
        print(f"Cached to {cache_filename}")
        return data
    except Exception as e:
        raise ConnectionError(f"Failed to fetch market data: {e}")


# ==============================================================================
# 4. STATISTICAL TOOLKIT (unchanged from v4; already correct on verification)
# ==============================================================================

def adf_test(series, name=""):
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, pvalue, usedlag, nobs, crit, icbest = adfuller(series.dropna(), autolag='AIC')
        return {"Series": name, "ADF_stat": stat, "p_value": pvalue,
                "lags_used": usedlag, "n_obs": nobs,
                "crit_1pct": crit['1%'], "crit_5pct": crit['5%'], "crit_10pct": crit['10%']}
    except ImportError:
        return {"Series": name, "ADF_stat": np.nan, "p_value": np.nan,
                "note": "statsmodels not installed; run `pip install statsmodels`."}


def ljung_box_manual(x, lags=(10, 20)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    xc = x - x.mean()
    acf_all = np.correlate(xc, xc, mode='full')[n - 1:] / np.arange(n, 0, -1)
    acf_all = acf_all / acf_all[0]
    rows = []
    for h in lags:
        rho = acf_all[1:h + 1]
        Q = n * (n + 2) * np.sum((rho ** 2) / (n - np.arange(1, h + 1)))
        p = 1 - stats.chi2.cdf(Q, df=h)
        rows.append({"lag": h, "Q_stat": Q, "p_value": p})
    return pd.DataFrame(rows)


def historical_var_es(returns, alpha=0.05):
    r = returns.dropna()
    var = -np.quantile(r, alpha)
    tail = r[r <= -var]
    es = -tail.mean() if len(tail) > 0 else np.nan
    return var, es


def parametric_var_cf(returns, alpha=0.05):
    r = returns.dropna()
    mu, sigma = r.mean(), r.std()
    S = stats.skew(r)
    K = stats.kurtosis(r, fisher=True)
    z = stats.norm.ppf(alpha)
    z_cf = z + (z**2 - 1) * S / 6 + (z**3 - 3*z) * K / 24 - (2*z**3 - 5*z) * (S**2) / 36
    return -(mu + z_cf * sigma)


def evt_var_es(returns, alpha=0.05, threshold_q=0.95):
    losses = -returns.dropna().values
    u = np.quantile(losses, threshold_q)
    exceed = losses[losses > u] - u
    n = len(losses)
    Nu = len(exceed)
    if Nu < 20:
        return {"VaR": np.nan, "ES": np.nan, "xi": np.nan, "beta": np.nan, "u": u, "Nu": Nu,
                "note": "Insufficient tail exceedances (<20) for a reliable GPD fit."}
    xi, _, beta = stats.genpareto.fit(exceed, floc=0)
    if abs(xi) < 1e-6:
        var = u - beta * np.log((n / Nu) * alpha)
        es = var + beta
    else:
        var = u + (beta / xi) * (((n / Nu) * alpha) ** (-xi) - 1)
        es = (var + beta - xi * u) / (1 - xi)
    return {"VaR": var, "ES": es, "xi": xi, "beta": beta, "u": u, "Nu": Nu}


def kupiec_pof_test(hit_series, alpha):
    hits = np.asarray(hit_series)
    n = len(hits)
    x = int(hits.sum())
    p = alpha
    p_hat = x / n if n > 0 else np.nan
    if x == 0 or x == n or p_hat in (0, 1):
        return {"LR_stat": np.nan, "p_value": np.nan, "n": n, "violations": x, "expected": p * n}
    log_num = (n - x) * np.log(1 - p) + x * np.log(p)
    log_den = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr = -2 * (log_num - log_den)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR_stat": lr, "p_value": p_value, "n": n, "violations": x, "expected": p * n}


def christoffersen_independence_test(hit_series):
    hits = np.asarray(hit_series).astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(hits)):
        prev, curr = hits[i - 1], hits[i]
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        elif prev == 1 and curr == 1: n11 += 1
    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0

    def _safe(p, x):
        return x * np.log(p) if (p > 0 and x > 0) else 0

    ll_r = _safe(pi, n01) + _safe(1 - pi, n00) + _safe(pi, n11) + _safe(1 - pi, n10)
    ll_u = _safe(pi01, n01) + _safe(1 - pi01, n00) + _safe(pi11, n11) + _safe(1 - pi11, n10)
    lr_ind = -2 * (ll_r - ll_u)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    return {"LR_ind": lr_ind, "p_value": p_value}


def backtest_var(returns, alpha=0.05, window=250):
    returns = returns.dropna()
    hits, idx = [], []
    for t in range(window, len(returns) - 1):
        train = returns.iloc[t - window:t]
        var_t = -np.quantile(train, alpha)
        realized = returns.iloc[t + 1]
        hits.append(1 if realized < -var_t else 0)
        idx.append(returns.index[t + 1])
    hit_series = pd.Series(hits, index=idx)
    kupiec = kupiec_pof_test(hit_series, alpha)
    christoffersen = christoffersen_independence_test(hit_series)
    return hit_series, kupiec, christoffersen


# ==============================================================================
# 5. GARCH MODEL RACE (unchanged from v4)
# ==============================================================================

def run_garch_model_race(returns_pct, asset_name):
    if not HAS_ARCH:
        print("  [ERROR] `arch` is not installed. Run: pip install arch")
        return None, None, None

    specs = [
        ("GARCH", dict(vol='Garch', p=1, q=1)),
        ("EGARCH", dict(vol='EGARCH', p=1, q=1)),
        ("GJR-GARCH", dict(vol='Garch', p=1, o=1, q=1)),
    ]
    dists = ["normal", "t", "skewt"]
    rows, fitted = [], {}
    for name, kwargs in specs:
        for dist in dists:
            try:
                am = arch_model(returns_pct.dropna(), dist=dist, **kwargs)
                res = am.fit(disp='off', show_warning=False)
                rows.append({"Model": name, "Distribution": dist,
                              "LogLik": res.loglikelihood, "AIC": res.aic, "BIC": res.bic})
                fitted[(name, dist)] = res
            except Exception:
                rows.append({"Model": name, "Distribution": dist,
                              "LogLik": np.nan, "AIC": np.nan, "BIC": np.nan})

    comp_df = pd.DataFrame(rows).sort_values("BIC")
    if comp_df["BIC"].isna().all():
        print("  [ERROR] All GARCH specifications failed to converge.")
        return None, comp_df, None

    best_model, best_dist = comp_df.iloc[0][["Model", "Distribution"]]
    best_res = fitted[(best_model, best_dist)]
    std_resid = (best_res.resid / best_res.conditional_volatility).dropna()

    lb_levels = ljung_box_manual(std_resid.values, lags=(10, 20))
    lb_squares = ljung_box_manual((std_resid.values) ** 2, lags=(10, 20))

    print(f"\n--- GARCH Model Race: {asset_name} ---")
    print(comp_df.to_string(index=False))
    print(f"\nSelected by BIC: {best_model} ({best_dist})")
    print("\nLjung-Box on standardized residuals:")
    print(lb_levels.to_string(index=False))
    print("\nLjung-Box on SQUARED standardized residuals:")
    print(lb_squares.to_string(index=False))

    export_table(comp_df, f"table_garch_race_{asset_name.lower().replace(' ', '_')}",
                 caption=f"GARCH-family model comparison for {asset_name} log returns (ranked by BIC).",
                 label=f"tab:garch_{asset_name.lower()}")

    diagnostics = {"levels": lb_levels, "squares": lb_squares,
                   "model": best_model, "dist": best_dist}
    return best_res, comp_df, diagnostics


# ==============================================================================
# 6. DCC-GARCH + [NEW IN v5] MACRO-LIQUIDITY PARTIAL CORRELATION CONTROL
# ==============================================================================

def dcc_garch_bivariate(returns_df, asset_names, max_iter=400):
    """Two-step DCC-GARCH(1,1) (Engle, 2002). Unchanged from v4."""
    if not HAS_ARCH:
        print("  [ERROR] `arch` is not installed. Run: pip install arch")
        return None, None

    std_resid = {}
    for col in asset_names:
        am = arch_model(returns_df[col].dropna(), vol='Garch', p=1, q=1, dist='t')
        res = am.fit(disp='off')
        std_resid[col] = res.resid / res.conditional_volatility
    Z = pd.concat(std_resid, axis=1).dropna()
    Z.columns = asset_names
    z = Z.values
    T, N = z.shape
    Qbar = np.cov(z.T)

    def unpack(theta):
        ra, rb = theta
        a = 1 / (1 + np.exp(-ra)) * 0.3
        b = 1 / (1 + np.exp(-rb)) * 0.95
        if a + b >= 0.999:
            b = 0.999 - a
        return a, b

    def neg_loglik(theta):
        a, b = unpack(theta)
        Qt = Qbar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                zt1 = z[t - 1].reshape(-1, 1)
                Qt = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Qt
            d = np.sqrt(np.diag(Qt))
            Rt = Qt / np.outer(d, d)
            try:
                sign, logdet = np.linalg.slogdet(Rt)
                if sign <= 0:
                    return 1e10
                Rinv = np.linalg.inv(Rt)
            except np.linalg.LinAlgError:
                return 1e10
            zt = z[t].reshape(-1, 1)
            ll += 0.5 * (logdet + (zt.T @ Rinv @ zt).item())
        return ll

    opt = minimize(neg_loglik, x0=np.array([0.0, 2.0]), method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    a, b = unpack(opt.x)

    Qt = Qbar.copy()
    corr_series = []
    for t in range(T):
        if t > 0:
            zt1 = z[t - 1].reshape(-1, 1)
            Qt = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Qt
        d = np.sqrt(np.diag(Qt))
        Rt = Qt / np.outer(d, d)
        corr_series.append(Rt[0, 1])
    dcc = pd.Series(corr_series, index=Z.index, name=f"DCC_{asset_names[0]}_{asset_names[1]}")
    print(f"DCC-GARCH(1,1) estimated: a={a:.4f}, b={b:.4f}, persistence(a+b)={a + b:.4f}")
    return dcc, {"a": a, "b": b}


def rolling_partial_correlation(returns_df, x_col, y_col, control_cols, window=60):
    """
    [NEW IN v5 -- addresses the audit's DCC-GARCH-omitted-confounders point]
    Rolling PARTIAL Pearson correlation between x_col and y_col, controlling
    linearly for control_cols, computed window-by-window via the standard
    residual-regression method: regress x and y each on the controls within
    the window, then correlate the residuals. This nets out shared linear
    co-movement with e.g. the Dollar Index and the VIX before measuring the
    Bitcoin-equity relationship, directly testing whether the raw DCC-GARCH
    correlation is a macro-liquidity artifact or survives the confound.
    """
    df = returns_df[[x_col, y_col] + control_cols].dropna()
    idx = df.index
    out = pd.Series(index=idx, dtype=float)
    X = df[control_cols].values
    x = df[x_col].values
    y = df[y_col].values
    n = len(df)
    for t in range(window, n):
        Xw = X[t - window:t]
        Xw1 = np.column_stack([np.ones(len(Xw)), Xw])
        xw = x[t - window:t]
        yw = y[t - window:t]
        try:
            bx, *_ = np.linalg.lstsq(Xw1, xw, rcond=None)
            by, *_ = np.linalg.lstsq(Xw1, yw, rcond=None)
            resid_x = xw - Xw1 @ bx
            resid_y = yw - Xw1 @ by
            if resid_x.std() > 0 and resid_y.std() > 0:
                out.iloc[t] = np.corrcoef(resid_x, resid_y)[0, 1]
        except np.linalg.LinAlgError:
            continue
    return out


# ==============================================================================
# 7. FIGURE FUNCTIONS
# ==============================================================================

def fig01_hierarchy_of_money():
    print("\nGenerating Figure 1: Hierarchy of Money (conceptual diagram)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10); ax.set_ylim(1.6, 9.2)
    ax.axis('off')
    tiers = [
        ("Sovereign Fiat Currency\n(state's own IOU; tax-enforced acceptance)", 8.2, CB["vermillion"]),
        ("Central Bank Reserves / Settlement Balances", 6.4, CB["orange"]),
        ("Commercial Bank Deposits", 4.6, CB["blue"]),
        ("Private Debt Instruments\n(corporate/household IOUs)", 2.8, CB["grey"]),
    ]
    box_w, box_h = 7.0, 1.3
    for label, y, color in tiers:
        ax.add_patch(mpatches.FancyBboxPatch((1.5, y - box_h/2), box_w, box_h,
                     boxstyle="round,pad=0.08", facecolor=color, alpha=0.85, edgecolor='black'))
        ax.text(5.0, y, label, ha='center', va='center', fontsize=10.5, weight='bold', color='white')
    ax.annotate('', xy=(0.9, 8.2), xytext=(0.9, 2.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(0.5, 5.5, 'Increasing acceptability\n& liquidity', rotation=90, ha='center', va='center', fontsize=9)
    ax.set_title('The Post-Keynesian "Hierarchy of Money" (Fig. 1)\nCONCEPTUAL DIAGRAM', fontsize=13)
    footnote(fig, "Pure conceptual illustration (Minsky, 1986; Wray, 2015); not derived from data.")
    plt.tight_layout()
    save_fig(fig, "figure_01_hierarchy_of_money")
    plt.close(fig)


def fig02_volatility_comparison(df_raw):
    print("\nGenerating Figure 2: Comparative Rolling Volatility...")
    windows = [15, 200]
    df = df_raw.rename(columns=ASSETS_FOR_VOL_COMP).dropna()
    colors = {'Apple': CB["sky"], 'Bitcoin': CB["orange"], 'Gold': CB["vermillion"]}
    for col in ASSETS_FOR_VOL_COMP.values():
        df[f'{col} Returns'] = df[col].pct_change()
        ann = np.sqrt(TRADING_DAYS.get(col, 252))
        for w in windows:
            df[f'{col} Volatility {w}d'] = df[f'{col} Returns'].rolling(w).std() * ann
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)
    for w, ax, label in zip(windows, axes, ["Short-Term (15-Day)", "Long-Term (200-Day)"]):
        for name in ASSETS_FOR_VOL_COMP.values():
            df[f'{name} Volatility {w}d'].plot(ax=ax, color=colors[name], lw=1.8, label=name)
        ax.set_ylabel(f"{label} Ann. Volatility")
        ax.legend(loc="upper left")
    fig.suptitle("Comparative Rolling Volatility: Bitcoin, Gold, Apple (Fig. 2)", fontsize=15)

    footnote(fig, "Annualization: 365 days for Bitcoin (24/7 market), 252 for equities/gold.")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "figure_02_rolling_volatility")
    plt.close(fig)
    return df


def fig03_04_risk_suite(data):
    print("\nGenerating Figures 3-4: VaR/ES Suite, Backtests, and GARCH Model Race...")
    price_cols = [TICKERS[k] for k in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']]
    log_returns = np.log(data[price_cols] / data[price_cols].shift(1)).dropna() * 100

    rows = []
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        ticker = TICKERS[name]
        if ticker not in log_returns.columns:
            continue
        r = log_returns[ticker]
        hist_var, hist_es = historical_var_es(r, 0.05)
        cf_var = parametric_var_cf(r, 0.05)
        evt = evt_var_es(r, 0.05)
        rows.append({
            "Asset": name, "Historical_VaR95": hist_var, "Historical_ES95": hist_es,
            "CornishFisher_VaR95": cf_var, "EVT_VaR95": evt["VaR"], "EVT_ES95": evt["ES"],
            "EVT_xi": evt.get("xi", np.nan),
        })
    var_df = pd.DataFrame(rows).sort_values("Historical_VaR95", ascending=False)
    print(var_df.to_string(index=False))
    export_table(var_df, "table_var_es_comparison",
                 caption="1-day 95% VaR and Expected Shortfall by method (percent).", label="tab:var_es")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(var_df))
    width = 0.25
    ax.bar(x - width, var_df["Historical_VaR95"], width, label="Historical VaR", color=CB["blue"])
    ax.bar(x, var_df["CornishFisher_VaR95"], width, label="Cornish-Fisher VaR", color=CB["orange"])
    ax.bar(x + width, var_df["EVT_VaR95"], width, label="EVT (POT/GPD) VaR", color=CB["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels(var_df["Asset"])
    ax.set_ylabel("1-Day 95% VaR (%)")
    ax.set_title("1-Day 95% VaR by Method (Fig. 3)")

    ax.legend()
    footnote(fig, "EVT VaR uses a Peaks-Over-Threshold GPD fit (McNeil & Frey, 2000); "
                   "Cornish-Fisher adjusts for skewness/kurtosis.")
    plt.tight_layout()
    save_fig(fig, "figure_03_var_comparison")
    plt.close(fig)

    bt_rows = []
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        ticker = TICKERS[name]
        if ticker not in log_returns.columns:
            continue
        r = log_returns[ticker]
        if len(r) < 300:
            continue
        hit_series, kupiec, christoffersen = backtest_var(r, alpha=0.05, window=250)
        bt_rows.append({
            "Asset": name, "N_obs": kupiec["n"], "Violations": kupiec["violations"],
            "Expected": kupiec["expected"], "Kupiec_LR": kupiec["LR_stat"],
            "Kupiec_p": kupiec["p_value"], "Christoffersen_LR": christoffersen["LR_ind"],
            "Christoffersen_p": christoffersen["p_value"],
        })
    bt_df = pd.DataFrame(bt_rows)
    print("\nRolling out-of-sample VaR backtests:")
    print(bt_df.to_string(index=False))
    export_table(bt_df, "table_var_backtests",
                 caption="Kupiec and Christoffersen VaR backtests.", label="tab:var_backtest")

    btc_res, garch_comp, garch_diag = run_garch_model_race(log_returns[TICKERS['Bitcoin']], "Bitcoin")
    if btc_res is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(log_returns.index, log_returns[TICKERS['Bitcoin']], color=CB["grey"], alpha=0.6,
                 lw=0.8, label='Daily Log Return (%)')
        ax2.plot(btc_res.conditional_volatility.index, btc_res.conditional_volatility,
                 color=CB["vermillion"], lw=1.6,
                 label=f'Conditional Volatility ({garch_diag["model"]}-{garch_diag["dist"]}, BIC-selected)')
        ax2.set_title('Bitcoin Returns and BIC-Selected Conditional Volatility (Fig. 4)')
        ax2.set_ylabel('Percent (%)')
        ax2.legend()
        footnote(fig2, "Model selected from a 9-specification race by BIC; see "
                        "table_garch_race_bitcoin.csv for full comparison and diagnostics.")
        plt.tight_layout()
        save_fig(fig2, "figure_04_garch_volatility")
        plt.close(fig2)

    return var_df, bt_df, garch_comp


def fig05_settlement_layer_comparison():
    """
    [UPDATED IN v5] Fedwire TPS corrected using the Fed's own 2024 PFMI
    disclosure (836,322 avg daily transactions -> ~9.68 TPS), replacing v4's
    6.65 TPS figure, which was computed from an annual count that could not
    be re-verified. T2 is now shown as a RANGE (400k-450k payments/day) since
    the ECB's TARGET Services Annual Report 2024 does not publish a single
    clean annual transaction count the way Fedwire does; v4's precise
    107,999,982 figure is dropped rather than repeated without a live source.
    """
    print("\nGenerating Figure 5: Settlement-Layer-Matched Throughput Comparison (v5, corrected)...")
    seconds_in_year = 365.25 * 24 * 60 * 60
    btc_tps = 6.5

    fedwire_avg_daily_txns = 836_322  # Fed 2024 PFMI disclosure
    fedwire_tps = fedwire_avg_daily_txns / 86400.0

    t2_daily_low, t2_daily_high = 400_000, 450_000  # order-of-magnitude, disclosed as uncertain
    t2_tps_low, t2_tps_high = t2_daily_low / 86400.0, t2_daily_high / 86400.0
    t2_tps_mid = (t2_tps_low + t2_tps_high) / 2

    mastercard_tps = 159.4e9 / seconds_in_year
    visa_tps = 303e9 / seconds_in_year

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    ax1 = axes[0]
    systems1 = ['Bitcoin L1\n(Settlement)', 'Fedwire\n(RTGS, USD)', 'T2\n(RTGS, EUR)\n[range]']
    vals1 = [btc_tps, fedwire_tps, t2_tps_mid]
    errs1 = [0, 0, (t2_tps_high - t2_tps_low) / 2]
    bars1 = ax1.bar(systems1, vals1, yerr=errs1, capsize=6,
                     color=[CB["orange"], CB["blue"], CB["sky"]])
    ax1.set_ylim(0, max(vals1) * 1.5)
    ax1.set_ylabel('Transactions Per Second (linear scale)')
    ax1.set_title('Settlement Layer:\nBitcoin L1 vs. RTGS Systems', fontsize=12)
    for b, v in zip(bars1, vals1):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f'{v:,.2f}',
                  ha='center', va='bottom', fontsize=9, weight='bold')

    ax2 = axes[1]
    systems2 = ['Bitcoin L1\n(Settlement)', 'Mastercard\n(Retail Auth.)', 'Visa\n(Retail Auth.)']
    vals2 = [btc_tps, mastercard_tps, visa_tps]
    bars2 = ax2.bar(systems2, vals2, color=[CB["orange"], CB["vermillion"], CB["purple"]])
    ax2.set_yscale('log')
    ax2.set_ylabel('Transactions Per Second (log scale)')
    ax2.set_title('Retail Layer:\nBitcoin L1 vs. Card-Network Authorization', fontsize=12)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height(), f'{b.get_height():,.0f}',
                  ha='center', va='bottom', fontsize=9, weight='bold')

    fig.suptitle('Transaction Throughput by Settlement Layer, Corrected (Fig. 5)', fontsize=15)
    footnote(fig, "Fedwire TPS corrected in v5 using the Fed's own 2024 PFMI disclosure "
                   "(836,322 avg. daily transactions). T2's exact 2024 transaction count is not "
                   "cleanly published; shown here as an order-of-magnitude RANGE with that "
                   "uncertainty disclosed, not a false-precision point estimate. This figure "
                   "supports a forced-choice (trilemma) argument, not a claim that either panel "
                   "alone settles whether Bitcoin's throughput is adequate -- see paper Sec. 4.")
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_fig(fig, "figure_05_settlement_layer_comparison")
    plt.close(fig)


def fig06_ln_reliability_synthesis():
    """
    [REPLACES v4's fig06] v4 showed ONLY Waugh & Holz (2020) with no context.
    v5 shows the 2020 mesh-probe numbers AND River Financial's 2023 single-hub
    platform success rate side by side, with an explicit annotation on why
    both are true and why the second does not refute the first -- it is the
    centralization outcome Avarikioti et al. (2020) predict.
    """
    print("\nGenerating Figure 6: LN Reliability Synthesis (2020 mesh probe vs. 2023 hub reliability)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    amounts = ['$0.01', '$10', '$50']
    success = [72, 44.15, 30.93]
    bars = ax1.bar(amounts, success, color=[CB["green"], CB["orange"], CB["vermillion"]])
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Payment Amount')
    ax1.set_ylabel('Routing Success Rate (%)')
    ax1.set_title('2020: Mesh-Wide Active Probe\n(Waugh & Holz, 4,626 nodes)', fontsize=11)
    for b, v in zip(bars, success):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:g}%', ha='center', fontsize=10, weight='bold')

    labels2 = ['River platform\n(Aug 2023, 308k txns)']
    vals2 = [99.7]
    bars2 = ax2.bar(labels2, vals2, color=CB["blue"], width=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Payment Success Rate (%)')
    ax2.set_title('2023: Single Well-Capitalized\nCustodial Hub', fontsize=11)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:g}%', ha='center', fontsize=11, weight='bold')

    fig.suptitle('Lightning Network Reliability: Mesh vs. Hub (Fig. 6)', fontsize=14)
    footnote(fig, "The right panel is NOT a like-for-like re-measurement of the left panel: it "
                   "reports one large, well-connected custodial node's own platform success rate, "
                   "not decentralized mesh-wide reachability. The gap between the two is the "
                   "predicted signature of the hub-and-spoke centralization formalized by "
                   "Avarikioti et al. (2020) -- reliability gains arrived via centralization, not "
                   "via a more reliable decentralized mesh. Sources: Waugh & Holz (2020); River "
                   "Financial (2023).")
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    save_fig(fig, "figure_06_ln_reliability_synthesis")
    plt.close(fig)


def fig07_ln_topology_schematic():
    print("\nGenerating Figure 7: LN Topology, Mesh vs. Hub-and-Spoke (conceptual schematic)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    n = 9
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs, ys = np.cos(angles), np.sin(angles)
    for ax, title in [(ax1, 'Early Stage: Mesh-Like'), (ax2, 'Mature Stage: Hub-and-Spoke')]:
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontweight='bold')
    for i in range(n):
        for j in range(i+1, n):
            if (i + j) % 2 == 0:
                ax1.plot([xs[i], xs[j]], [ys[i], ys[j]], color='#999', lw=0.8, zorder=1)
    ax1.scatter(xs, ys, s=140, color=CB["green"], edgecolor='black', zorder=2)
    for i in range(1, n):
        ax2.plot([0, xs[i]], [0, ys[i]], color='#999', lw=1.2, zorder=1)
    ax2.scatter(xs[1:], ys[1:], s=100, color=CB["green"], edgecolor='black', zorder=2)
    ax2.scatter([0], [0], s=320, color=CB["vermillion"], edgecolor='black', zorder=3)
    ax2.text(0, -1.3, 'Hub (liquidity-concentrated node)', ha='center', fontsize=8.5)
    fig.suptitle('Predicted Topological Evolution of the Lightning Network (Fig. 7)\n'
                 'CONCEPTUAL SCHEMATIC, not a simulation or measurement', fontsize=12)
    footnote(fig, "Deterministic, hand-placed schematic illustrating the mesh-to-hub-and-spoke "
                   "claim formalized by Avarikioti et al. (2020).")
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    save_fig(fig, "figure_07_ln_topology_schematic")
    plt.close(fig)


def fig08_mv_py_mismatch():
    print("\nGenerating Figure 8: Fixed-Supply / Growing-Output Mismatch...")
    years = np.arange(0, 21)
    M = np.full_like(years, 100.0, dtype=float)
    Y = 100.0 * (1.03) ** years
    V = 1.0
    P = (M * V) / Y
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    ax1.plot(years, M, color=CB["blue"], lw=2.2, label='Money Supply (M), fixed')
    ax1.plot(years, Y, color=CB["green"], lw=2.2, label='Real Output (Y), 3%/yr growth')
    ax1.set_xlabel('Years'); ax1.set_ylabel('Index (Year 0 = 100)')
    ax1.set_title('Fixed M vs. Growing Y'); ax1.legend()
    ax2.plot(years, P, color=CB["vermillion"], lw=2.2)
    ax2.set_xlabel('Years'); ax2.set_ylabel('Implied Price Level (P), Year 0 = 100')
    ax2.set_title('Implied Price Level from M\u00b7V = P\u00b7Y')
    fig.suptitle('Fixed-Supply vs. Growing-Output Mismatch (Fig. 8)\nILLUSTRATIVE IDENTITY', fontsize=13)
    footnote(fig, "Deterministic illustration of M*V=P*Y under fixed money supply, constant "
                   "velocity, and 3%/year output growth -- not a measurement of Bitcoin's actual M,V,P.")
    plt.tight_layout()
    save_fig(fig, "figure_08_mv_py_mismatch")
    plt.close(fig)


def fig09_debt_burden_deflation(annual_deflation=0.03, years=30):
    """
    [ENHANCED IN v5] Left panel unchanged from v4 (correct deterministic
    calculation). NEW right panel illustrates the benign-vs-malignant
    deflation distinction the paper's text now makes explicit: the 1920-21
    vs. 1929-30 comparison, where the same direction of price change had
    opposite real consequences depending on pre-existing private leverage.
    This is a schematic/illustrative panel (stylized, not a fitted historical
    reconstruction) making a qualitative point already sourced in the text
    to Dimand (1994).
    """
    print(f"\nGenerating Figure 9: Real Debt Burden Under Deflation + Malignant/Benign Distinction...")
    t = np.arange(0, years + 1)
    real_burden_multiple = (1 - annual_deflation) ** (-t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(t, real_burden_multiple, color=CB["vermillion"], lw=2.4)
    ax1.fill_between(t, 1, real_burden_multiple, color=CB["vermillion"], alpha=0.15)
    ax1.axhline(1, color='black', lw=0.8, ls='--')
    ax1.set_xlabel('Years Into a Fixed-Payment Debt Contract')
    ax1.set_ylabel('Real Burden Multiple (relative to Year 0)')
    ax1.set_title(f'Real Burden of a Fixed Nominal Payment\nUnder {annual_deflation:.0%}/yr Deflation', fontsize=11)
    end_val = real_burden_multiple[-1]
    ax1.annotate(f'{end_val:.2f}x after {years} years', xy=(years, end_val),
                xytext=(years - 8, end_val - 0.35), fontsize=9, weight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    years_stylized = np.array([0, 1, 2, 3])
    low_leverage = np.array([0, -6, -3, 1])   # 1920-21-style: sharp but brief
    high_leverage = np.array([0, -6, -18, -30])  # 1929-30-style: cascades via leverage
    ax2.plot(years_stylized, low_leverage, 'o-', color=CB["green"], lw=2.4,
             label='Low pre-existing leverage\n(1920-21 pattern)')
    ax2.plot(years_stylized, high_leverage, 's-', color=CB["vermillion"], lw=2.4,
             label='High pre-existing leverage\n(1929-30 pattern)')
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_xlabel('Years After Deflationary Shock Begins')
    ax2.set_ylabel('Stylized Real Output Path (%)')
    ax2.set_title('Same Deflation, Different Outcome:\nMechanism is Leverage, Not Sign', fontsize=11)
    ax2.legend(fontsize=8, loc='lower left')

    fig.suptitle('Debt-Deflation: Magnitude (left) and Mechanism (right) (Fig. 9)', fontsize=13)
    footnote(fig, f"Left: deterministic calculation real_burden(t)=(1-{annual_deflation:.2f})^(-t). "
                   "Right: STYLIZED, illustrative curves (not a fitted historical reconstruction) "
                   "making the qualitative point, sourced to Dimand (1994), that 1920-21 and "
                   "1929-30 were both deflations but diverged sharply because of pre-existing "
                   "private-sector leverage -- the mechanism this paper's deflation critique "
                   "actually depends on, not deflation's sign alone.")
    plt.tight_layout()
    save_fig(fig, "figure_09_debt_burden_deflation")
    plt.close(fig)


def fig10_11_digital_gold_narrative(data):
    print("\nGenerating Figure 10: Drawdowns...")
    btc = data[TICKERS['Bitcoin']].dropna()
    dd = (btc - btc.cummax()) / btc.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(btc.index, btc, color=CB["blue"], label='Bitcoin Price (USD)')
    ax1.set_yscale('log'); ax1.legend(); ax1.set_ylabel('Price (log scale)')
    ax1.set_title('Bitcoin Price and Historical Drawdowns (Fig. 10)')
    ax2.plot(dd.index, dd*100, color=CB["vermillion"])
    ax2.fill_between(dd.index, dd*100, 0, color=CB["vermillion"], alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    max_dd = dd.min()*100
    ax2.text(dd.idxmin(), max_dd, f'Max DD: {max_dd:.1f}%', ha='right', va='top', fontsize=9)
    plt.tight_layout()
    save_fig(fig, "figure_10_drawdowns")
    plt.close(fig)

    print("\nGenerating Figure 11: Rolling Correlation + DCC-GARCH + [NEW] Macro-Liquidity Control...")
    pair_cols = [TICKERS['Bitcoin'], TICKERS['S&P 500']]
    log_returns = np.log(data[pair_cols] / data[pair_cols].shift(1)).dropna() * 100
    rolling_corr = log_returns[TICKERS['Bitcoin']].rolling(60).corr(log_returns[TICKERS['S&P 500']])

    fig2, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(rolling_corr.index, rolling_corr, color=CB["grey"], alpha=0.5, lw=1.0,
            label='60-Day Rolling Pearson (naive baseline)')

    if HAS_ARCH:
        dcc, params = dcc_garch_bivariate(log_returns, pair_cols)
        if dcc is not None:
            ax.plot(dcc.index, dcc, color=CB["vermillion"], lw=1.6,
                    label=f'DCC-GARCH(1,1), raw (a={params["a"]:.3f}, b={params["b"]:.3f})')

        macro_cols = [TICKERS['Bitcoin'], TICKERS['S&P 500'], TICKERS['US Dollar'], TICKERS['VIX']]
        if all(c in data.columns for c in macro_cols):
            macro_returns = np.log(data[macro_cols] / data[macro_cols].shift(1)).dropna() * 100
            partial = rolling_partial_correlation(
                macro_returns, TICKERS['Bitcoin'], TICKERS['S&P 500'],
                [TICKERS['US Dollar'], TICKERS['VIX']], window=60)
            ax.plot(partial.index, partial, color=CB["blue"], lw=1.6, ls='--',
                    label='60-Day Partial Correlation\n(net of DXY & VIX co-movement)')
        else:
            print("  [WARN] VIX or Dollar Index data unavailable; partial-correlation panel skipped.")
    else:
        print("  [WARN] `arch` not installed; DCC-GARCH panel skipped.")

    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.set_title('Bitcoin vs. S&P 500: Raw Correlation vs. Macro-Liquidity-Controlled (Fig. 11)')
    ax.set_ylabel('Correlation')
    ax.legend(fontsize=9)
    footnote(fig2, "NEW IN v5: the dashed blue line nets out each 60-day window's linear "
                    "co-movement with the Dollar Index (UUP) and the VIX before computing the "
                    "Bitcoin-S&P 500 correlation, directly testing whether the raw DCC-GARCH "
                    "correlation spike during 2020/2022 stress periods is a shared macro-liquidity "
                    "artifact or survives the confound. Where the dashed line stays well below the "
                    "solid red line during stress windows, the macro-confound critique has merit; "
                    "where it does not, the 'Bitcoin behaves like risk-on tech' reading survives it.")
    plt.tight_layout()
    save_fig(fig2, "figure_11_correlation_dcc")
    plt.close(fig2)


def fig12_attack_cost_breakdown():
    """
    [UPDATED IN v5] v4 showed only Harvey's Oct-2025 $6B static estimate. v5
    shows BOTH the original 2025 estimate and Harvey's own July-2026 update to
    ~$8B, which crucially adds a derivatives-shorting profit mechanism -- the
    threat model got MORE credible on the latest public evidence, not less,
    contrary to the "static cost, no motive" line of argument.
    """
    print("\nGenerating Figure 12: 51% Attack Cost, 2025 vs. 2026 Estimates...")
    components_2025 = {'Hardware\n(ASICs)': 4.6, 'Data Center\nCapEx': 1.34, 'Energy\n(1 week)': 0.13}
    total_2025 = sum(components_2025.values())
    total_2026 = 8.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    bottom = 0
    for (label, val), color in zip(components_2025.items(), [CB["blue"], CB["sky"], CB["orange"]]):
        ax1.bar(['Oct 2025\nestimate'], [val], bottom=bottom, label=f'{label} (${val:.2f}B)', color=color)
        bottom += val
    ax1.text(0, total_2025 + 0.15, f'Total: ${total_2025:.2f}B', ha='center', fontweight='bold')
    ax1.set_ylabel('Cost (USD, Billions)')
    ax1.set_title('Static Hardware/Infrastructure Cost\n(Harvey, Oct 2025)', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, total_2025 * 1.5)

    bars2 = ax2.bar(['Oct 2025\n(~$6B, static cost)', 'Jul 2026\n(~$8B, + derivatives\nshort profit motive)'],
                     [total_2025, total_2026], color=[CB["grey"], CB["vermillion"]])
    ax2.set_ylabel('Estimated Cost (USD, Billions)')
    ax2.set_title('The Estimate Moved Up, Not Down,\nOnce a Profit Motive Was Added', fontsize=11)
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, f'${b.get_height():,.1f}B',
                  ha='center', va='bottom', fontsize=10, weight='bold')

    fig.suptitle("51% Attack Economics: Harvey's Own Revision (Fig. 12)", fontsize=14)
    footnote(fig, "REPRODUCED FROM CITED SOURCES: Harvey (Oct. 2025) estimated ~$6B in static "
                   "hardware/infrastructure cost for a one-week attack. Harvey's own July 2026 "
                   "update raises this to ~$8B and adds that combining majority hashpower with a "
                   "large short position in Bitcoin derivatives makes the attack a rational, "
                   "profit-motivated trade rather than a costly act with no return -- a mechanism "
                   "the static hardware-cost accounting omits entirely. Neither estimate accounts "
                   "for hardware-supply inelasticity or grid-procurement lead times, which would "
                   "push the effective cost higher still; neither accounts for the possibility of a "
                   "defensive proof-of-work fork, which would push the attacker's realized payoff "
                   "lower. Both estimates should be read as bounds on a moving target, not as a "
                   "single settled number.")
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    save_fig(fig, "figure_12_attack_cost_breakdown")
    plt.close(fig)


def fig13_mining_pool_concentration():
    """
    [UPDATED IN v5] v4 showed a single point estimate (43%/65%) as of Nov
    2025. v5 shows a RANGE reflecting real disagreement/volatility across
    independent trackers (Hashrate Index, B10C, Spark) in 2025-2026, and adds
    a note on Stratum V2 adoption as a partial, real mitigation of pool-level
    censorship risk specifically (not of hashrate concentration itself).
    """
    print("\nGenerating Figure 13: Mining Pool Concentration (range across sources, 2025-2026)...")
    labels = ['Top 2 Pools\n(Foundry USA + AntPool)', 'Top 4-5 Pools\n(cumulative)']
    low = [43, 65]
    high = [57, 76]
    mid = [(l+h)/2 for l, h in zip(low, high)]
    err = [(h-l)/2 for l, h in zip(low, high)]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.barh(labels, mid, xerr=err, capsize=8, color=[CB["vermillion"], CB["orange"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Cumulative Share of Network Hashrate (%), range across trackers')
    ax.set_title('Bitcoin Mining Pool Concentration, 2025-2026 (Fig. 13)')
    for b, l, h in zip(bars, low, high):
        ax.text(h + 2, b.get_y() + b.get_height()/2, f'{l}-{h}%', va='center', fontsize=10, weight='bold')
    ax.text(2, -0.85, 'Stratum V2 (adopted by pools representing ~75% of hashrate as of 2026) lets '
                       'individual miners build their own block templates,\nmitigating pool-level '
                       'transaction censorship specifically -- it does NOT reduce hashrate '
                       'concentration or 51%-reorg risk.', fontsize=8, style='italic', color='#444444',
                       transform=ax.transData)
    footnote(fig, "REPRODUCED AS A RANGE across independent trackers (Hashrate Index; B10C, 2025; "
                   "Spark, 2026), since single-point estimates for this metric swing by several "
                   "points week to week and no single tracker should be presented as definitive.")
    plt.tight_layout()
    save_fig(fig, "figure_13_mining_pool_concentration")
    plt.close(fig)


def fig14_security_budget_dilemma():
    print("\nGenerating Figure 14: Security Budget Dilemma (theoretical model)...")
    fig, ax = plt.subplots(figsize=(11, 7.5))
    x_events = {"Present": 0, "2028 Halving": 1.5, "2032 Halving": 3.0, "Post-Subsidy Era": 4.5}
    y_levels = {"Vulnerable": 0, "Low": 1, "Medium": 2, "High": 3}
    x_subsidy = [0,1.5,1.5,3.0,3.0,4.5,4.5,5.5]; y_subsidy = [3,3,2,2,1,1,0.4,0.2]
    x_l1 = [0,1.5,1.5,5.5]; y_l1 = [3,3,2.8,2.5]
    ax.plot(x_subsidy, y_subsidy, '--', color=CB["blue"], lw=2.3, label='Scenario A: High L2 Adoption')
    ax.plot(x_l1, y_l1, '-', color=CB["vermillion"], lw=2.3, label='Scenario B: L1 Fee Retention')
    ax.set_xticks(list(x_events.values())); ax.set_xticklabels(list(x_events.keys()))
    ax.set_yticks(list(y_levels.values())); ax.set_yticklabels(list(y_levels.keys()))
    ax.set_xlabel("Time"); ax.set_ylabel("Security Budget")
    ax.set_title("A Model of the Bitcoin Security-Budget Dilemma (Fig. 14)\nTHEORETICAL MODEL", fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig, "figure_14_security_budget_dilemma")
    plt.close(fig)


def fig15_governance_paralysis():
    print("\nGenerating Figure 15: Governance Models, Managed vs. Consensus (conceptual)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, title, color, bullets in [
        (ax1, 'Managed Monetary System', CB["blue"],
         ['Discretionary policy tools', 'Central decision authority',
          'Can respond to crises quickly', 'Adaptive to new information']),
        (ax2, "Bitcoin's Consensus Model", CB["vermillion"],
         ['Requires broad decentralized agreement', 'No central decision authority',
          'Prone to paralysis on contentious changes', 'Example: 2015-2017 "Blocksize Wars" split']),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0.5, 8.5); ax.axis('off')
        ax.add_patch(mpatches.FancyBboxPatch((0.5, 6), 9, 2, boxstyle="round,pad=0.1",
                     facecolor=color, alpha=0.85, edgecolor='black'))
        ax.text(5, 7, title, ha='center', va='center', fontsize=12.5, weight='bold', color='white')
        for i, b in enumerate(bullets):
            ax.text(1, 4.9 - i*1.15, f'\u2022 {b}', fontsize=10, va='center')
    fig.suptitle('Governance Models: Managed vs. Consensus-Based (Fig. 15)\nCONCEPTUAL DIAGRAM', fontsize=13)
    footnote(fig, "Conceptual comparison diagram; not derived from data.")
    plt.tight_layout(rect=[0, 0.05, 1, 0.85])
    save_fig(fig, "figure_15_governance_paralysis")
    plt.close(fig)


def fig16_wash_trading_updated():
    """
    [UPDATED IN v5] v4 showed only Bitwise (2019)'s 95% estimate. v5 adds
    Cong et al. (2022, NBER)'s more rigorous, more recent re-estimate and
    Sila et al. (2025), and the caption explicitly flags what none of these
    numbers describe: the now-larger regulated-venue (ETF/CME/MiCA) share of
    the market that has emerged since January 2024.
    """
    print("\nGenerating Figure 16: Wash-Trading Estimates, Updated Across Three Independent Studies...")
    studies = ['Bitwise (2019)\nSEC presentation', 'Cong et al. (2022)\nNBER, Benford/clustering tests',
               'Sila et al. (2025)\nvolatility-conditional estimate']
    low = [95, 70.85, 55]
    high = [95, 77.50, 85]
    mid = [(l+h)/2 for l, h in zip(low, high)]
    err = [(h-l)/2 for l, h in zip(low, high)]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(studies, mid, yerr=err, capsize=6,
                   color=[CB["vermillion"], CB["orange"], CB["purple"]])
    ax.set_ylabel('Estimated Wash-Trading Share of Reported\nVolume on UNREGULATED Exchanges (%)')
    ax.set_title('Wash-Trading Estimates on Unregulated Exchanges,\nThree Independent Studies (Fig. 16)')
    ax.set_ylim(0, 105)
    for b, l, h in zip(bars, low, high):
        label = f'{l:g}%' if l == h else f'{l:g}-{h:g}%'
        ax.text(b.get_x() + b.get_width()/2, h + 2, label, ha='center', fontsize=10, weight='bold')
    footnote(fig, "All three estimates describe UNREGULATED exchanges only. Since Jan. 2024, "
                   "U.S. spot Bitcoin ETF approval and MiCA implementation have shifted a "
                   "substantial share of institutional PRICE-FORMATION volume onto regulated venues "
                   "(CME futures, NYDFS/CFTC/SEC-supervised spot exchanges) not characterized by any "
                   "of these three studies; none of the bars above should be read as describing the "
                   "market's regulated tier.")
    plt.tight_layout()
    save_fig(fig, "figure_16_wash_trading_updated")
    plt.close(fig)


def fig17_tether_dominance():
    print("\nGenerating Figure 17: Tether Dominance of Stablecoin Market Cap (2025-2026 range)...")
    labels = ['Tether (USDT)', 'All Other Stablecoins']
    low, high = 58.3, 63.0
    vals = [(low + high) / 2, 100 - (low + high) / 2]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, vals, color=[CB["vermillion"], CB["grey"]])
    ax.set_ylabel('Share of Total Stablecoin Market Capitalization (%)')
    ax.set_title(f'Tether (USDT) Share of Stablecoin Market Cap (Fig. 17)\n(range {low:g}-{high:g}%, late 2025-2026)')
    ax.set_ylim(0, 70)
    ax.text(0, vals[0] + 1.5, f'~{low:g}-{high:g}%', ha='center', fontsize=11, weight='bold')
    footnote(fig, "Shown as a range across independent trackers (CoinMarketCap, DeFiLlama-based "
                   "aggregators) rather than a single archived figure, since USDT's exact share "
                   "moves within this band across late 2025 and 2026. The U.S. GENIUS Act (enacted "
                   "July 2025) establishes forward-looking federal reserve/redemption standards for "
                   "payment stablecoin issuers but does not retroactively resolve concentration risk "
                   "in Tether's existing market share.")
    plt.tight_layout()
    save_fig(fig, "figure_17_tether_dominance")
    plt.close(fig)


def fig18_elsalvador_impact():
    """
    [NEW IN v5] Reproduces Charfi (2024)'s published difference-in-differences
    point estimates for the macroeconomic impact of Bitcoin legal-tender
    adoption in El Salvador. This entire section (paper Sec. 10) was absent
    from v4 despite being the paper's single strongest available natural
    experiment.
    """
    print("\nGenerating Figure 18: El Salvador DiD Macroeconomic Impact (reproduced, cited)...")
    variables = ['GDP\nGrowth', 'Employment\nRate', 'Investment\nRate', 'Inflation\nRate',
                 'Remittance\nInflows', 'Bond\nYield']
    impacts = [0.78, -0.47, -0.65, 4.15, 1.81, 0.48]
    colors = [CB["green"] if v >= 0 and i not in (3, 5) else (CB["vermillion"] if v < 0 else CB["orange"])
              for i, v in enumerate(impacts)]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(variables, impacts, color=colors)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylim(-1, 5)  # Forces the y-axis to render negative ticks
    ax.set_ylabel('Estimated Impact (percentage points, DiD coefficient)')

    ax.set_title('Macroeconomic Impact of Bitcoin Legal-Tender Adoption\nin El Salvador (Fig. 18)')
    for b, v in zip(bars, impacts):
        ax.text(b.get_x() + b.get_width()/2, v + (0.15 if v >= 0 else -0.35), f'{v:+.2f}pp',
                 ha='center', fontsize=9, weight='bold')
    footnote(fig, "REPRODUCED FROM CITED SOURCE: Charfi (2024), difference-in-differences "
                   "estimates comparing El Salvador against a control group of neighboring "
                   "countries after Bitcoin's 2021 legal-tender adoption. Higher inflation and "
                   "bond yields plus lower employment and investment accompanied the policy; "
                   "remittance inflows rose modestly despite the policy's stated goal of a much "
                   "larger remittance-cost reduction.")
    plt.tight_layout()
    save_fig(fig, "figure_18_elsalvador_impact")
    plt.close(fig)

    df = pd.DataFrame({"Variable": variables, "DiD_Impact_pp": impacts})
    export_table(df, "table_el_salvador",
                 caption="Macroeconomic impact of Bitcoin adoption in El Salvador (Charfi, 2024).",
                 label="tab:elsalvador")


def fig19_basel_capital_requirement():
    print("\nGenerating Figure 19: Basel III Capital Requirement (deterministic, cited inputs)...")
    min_ratio = 0.08
    categories = {'Unbacked Crypto\n(BTC), RW=1250%\n(Group 2b, eff. Jan 2026)': 1250,
                  'Typical Unrated\nCorporate, RW=100%': 100}
    capital_pct = {k: (v / 100) * min_ratio * 100 for k, v in categories.items()}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    bars = ax.bar(list(capital_pct.keys()), list(capital_pct.values()),
                   color=[CB["vermillion"], CB["blue"]])
    ax.set_ylabel('Required Capital per $100 of Exposure ($)')
    ax.set_title('Basel III Capital Requirement by Exposure Class (Fig. 19)')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'${b.get_height():.0f}',
                 ha='center', va='bottom', fontsize=11, weight='bold')
    footnote(fig, "Deterministic calculation: capital = risk_weight x 8% minimum total capital "
                   "ratio. The 1250% Group 2b risk weight for unbacked crypto-assets is confirmed, "
                   "current, and binding as of January 1, 2026 (BCBS, 2022; Basel SCO60 standard).")
    plt.tight_layout()
    save_fig(fig, "figure_19_basel_capital_requirement")
    plt.close(fig)


def fig_stationarity_report(data):
    print("\nGenerating stationarity report (ADF tests on prices and returns)...")
    rows = []
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        ticker = TICKERS[name]
        if ticker not in data.columns:
            continue
        price = data[ticker].dropna()
        ret = np.log(price / price.shift(1)).dropna()
        rows.append(adf_test(price, f"{name} (level)"))
        rows.append(adf_test(ret, f"{name} (log return)"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    export_table(df, "table_adf_stationarity",
                 caption="Augmented Dickey-Fuller stationarity tests on price levels and log returns.",
                 label="tab:adf")
    return df


# ==============================================================================
# 8. MAIN MENU AND EXECUTION
# ==============================================================================

def main_menu(full_data, data_loaded):
    menu = {
        '1':  ('Fig 1: Hierarchy of money (conceptual)', fig01_hierarchy_of_money, None),
        '2':  ('Fig 2: Volatility comparison', fig02_volatility_comparison, 'full'),
        '3':  ('Figs 3-4: VaR/ES suite + backtests + GARCH race', fig03_04_risk_suite, 'full'),
        '4':  ('Fig 5: Settlement-layer throughput [v5: corrected Fedwire TPS]', fig05_settlement_layer_comparison, None),
        '5':  ('Fig 6: LN reliability synthesis [v5: NEW, replaces stale-only figure]', fig06_ln_reliability_synthesis, None),
        '6':  ('Fig 7: LN topology mesh vs. hub-and-spoke (schematic)', fig07_ln_topology_schematic, None),
        '7':  ('Fig 8: MV=PY mismatch (illustrative identity)', fig08_mv_py_mismatch, None),
        '8':  ('Fig 9: Debt burden [v5: + benign/malignant deflation panel]', fig09_debt_burden_deflation, None),
        '9':  ('Figs 10-11: Drawdowns + DCC-GARCH [v5: + macro-liquidity control]', fig10_11_digital_gold_narrative, 'full'),
        '10': ('Fig 12: 51% attack cost [v5: 2025 vs 2026 Harvey estimates]', fig12_attack_cost_breakdown, None),
        '11': ('Fig 13: Mining pool concentration [v5: range + Stratum V2 note]', fig13_mining_pool_concentration, None),
        '12': ('Fig 14: Security budget dilemma (theoretical)', fig14_security_budget_dilemma, None),
        '13': ('Fig 15: Governance models (schematic)', fig15_governance_paralysis, None),
        '14': ('Fig 16: Wash-trading [v5: 3 studies + institutional-era caveat]', fig16_wash_trading_updated, None),
        '15': ('Fig 17: Tether dominance [v5: range + GENIUS Act note]', fig17_tether_dominance, None),
        '16': ('Fig 18: El Salvador DiD impact [v5: NEW section]', fig18_elsalvador_impact, None),
        '17': ('Fig 19: Basel III capital requirement (deterministic, cited)', fig19_basel_capital_requirement, None),
        '18': ('Stationarity report (ADF, supplementary table)', fig_stationarity_report, 'full'),
        '19': ('Run all', 'run_all', None),
        '0':  ('Exit', 'exit', None),
    }

    def execute(choice):
        desc, func, needs = menu[choice]
        if needs == 'full' and not data_loaded:
            print(f"\n[ERROR] Cannot run '{desc}': market data unavailable.")
            return
        func(full_data) if needs == 'full' else func()

    def run_all():
        print("\n--- RUNNING ALL FIGURES/TABLES ---")
        for k in sorted(menu.keys(), key=int):
            if menu[k][1] not in ('run_all', 'exit'):
                try:
                    execute(k)
                except Exception as e:
                    print(f"  [ERROR] {menu[k][0]} failed: {e}")
        print("\n--- DONE ---")

    while True:
        print("\n" + "=" * 70)
        print("   v5 ANALYSIS -- AUDIT-RESPONSIVE REVISION -- MAIN MENU")
        print("=" * 70)
        for k in sorted(menu.keys(), key=int):
            print(f"  [{k}] {menu[k][0]}")
        print("-" * 70)
        choice = input("Enter your choice: ").strip()
        if choice == '0':
            break
        elif choice in menu:
            run_all() if menu[choice][1] == 'run_all' else execute(choice)
        else:
            print("Invalid choice.")
        if choice in menu and choice != '0':
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    print(f"Starting v5 analysis (audit-responsive revision, paper date: {FINAL_ANALYSIS_DATE})")
    print(f"Dependency status: yfinance={HAS_YFINANCE}, arch={HAS_ARCH}")
    if not HAS_YFINANCE or not HAS_ARCH:
        missing = [p for p, ok in [("yfinance", HAS_YFINANCE), ("arch", HAS_ARCH)] if not ok]
        print(f"  -> Missing: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")

    effective_end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    try:
        full_data = get_market_data(FULL_START_DATE, effective_end_date, CACHE_FILENAME)
        data_loaded = not full_data.empty
    except (ConnectionError, ValueError) as e:
        print(f"\n[CRITICAL] Could not load market data: {e}")
        full_data = pd.DataFrame()
        data_loaded = False

    main_menu(full_data, data_loaded)
