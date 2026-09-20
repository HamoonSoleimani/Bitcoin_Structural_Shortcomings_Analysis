#!/usr/bin/env python3
# ==============================================================================
#   REPRODUCIBLE ANALYSIS SCRIPT -- v7 (manuscript-scope realignment)
# ------------------------------------------------------------------------------
#   Title:   The Re-Intermediation Trilemma: Scaling, Liquidity, and
#            Structural Centralization in Bitcoin
#   Author:  Hamoon Soleimani
#
#   CHANGELOG vs. v6 (this revision realigns the script's outputs with a
#   narrowed manuscript: one thesis (structural re-intermediation),
#   evaluated through three connected empirical pillars, instead of a
#   dozen loosely-related critiques. Nothing here is "wrong" material being
#   discarded -- it is out-of-scope material being separated from the core
#   argument so the core argument is testable as ONE claim, not twelve.):
#
#   - Figures now match the paper's actual structure and numbering:
#       Fig. 1            NEW conceptual synthesis diagram (the trilemma
#                          itself: three layers, one mechanism)
#       Fig. 2             Settlement-layer throughput (was Fig. 5)
#       Fig. 3             LN topology schematic / theory (was Fig. 7)
#       Fig. 4             LN reliability synthesis / evidence (was Fig. 6)
#       Fig. 5             Security-budget dilemma (was Fig. 14)
#       Fig. 6             Mining-pool concentration (was Fig. 13)
#   - Moved to Appendix A (volatility/risk/correlation characterization --
#     genuine, correct analysis, but a different question from
#     re-intermediation): rolling volatility (was Fig. 2), VaR/ES suite and
#     GARCH race (was Figs. 3-4), drawdowns and DCC-GARCH correlation (was
#     Figs. 10-11), the ADF stationarity table.
#   - Moved to Appendix B (market-microstructure robustness, per the audit's
#     own suggestion to relocate this rather than cut it): wash-trading
#     estimates (was Fig. 16), stablecoin/Tether dominance (was Fig. 17).
#   - Moved to Appendix C (attack-cost economics, supplementary detail for
#     Pillar III rather than part of the core structural claim): the 51%
#     attack-cost breakdown (was Fig. 12).
#   - REMOVED (the audit's own "kitchen-sink" examples -- these are
#     legitimate topics but different papers, not different figures of this
#     one): the Post-Keynesian hierarchy-of-money diagram, the MV=PY
#     mismatch and debt-deflation figures, the governance-paralysis diagram,
#     the El Salvador difference-in-differences figure/table (Charfi 2024's
#     regression, not this paper's own -- the audit specifically flagged
#     presenting someone else's estimates as a full section as looking like
#     an unoriginal review), and the Basel III capital-requirement figure.
#   - Added `fig01_reintermediation_trilemma()`: a new conceptual diagram
#     that is the paper's central figure -- it did not exist before.
#   - `FIGURE_MANIFEST` gained a `Role` column (which section/appendix each
#     figure supports) alongside the existing `Type` column, so the
#     manifest documents both evidentiary status AND where each figure is
#     actually used.
#
#   ---------------------------------------------------------------------
#   CHANGELOG vs. v7 -> v7.1 (citation-precision pass, cross-checked against
#   a parallel review of the pre-restructuring paper lineage plus primary
#   sources located via live search; see the chat response for the full
#   accounting of what that parallel review got right vs. wrong):
#
#   - Stratum V2 chart annotation (Fig. 6) corrected: ~75% of hashrate is
#     WORKING-GROUP COMMITMENT (May 2026), not production deployment; only
#     ~3-5% of hashrate actually runs Stratum V2 with miner-selected
#     templates in production as of mid-2026. Both figures verified via live
#     search against multiple independent industry trackers. The old
#     annotation conflated the two, overstating the mitigation's current
#     real-world effect.
#
#   ---------------------------------------------------------------------
#   CHANGELOG vs. v5 -> v6 (retained below for provenance; still accurate --
#   this revision fixes every code-level issue raised in the editorial
#   audit, plus two additional correctness bugs found while fixing them):
#
#   [Audit Issue 1 -- calendar/annualization mismatch]
#     v5 loaded every ticker with a single joint `.dropna()`, which silently
#     intersected Bitcoin's calendar with equities/FX that never trade on
#     weekends -- truncating Bitcoin to ~252 obs/year while every downstream
#     figure still annualized it with sqrt(365). FIXED by loading every
#     ticker on its OWN native calendar (`load_all_series`) and performing
#     cross-asset date intersection in exactly ONE place, explicitly, only
#     for analyses that mathematically require paired same-day observations
#     (`build_aligned_panel`, used only for rolling correlation / DCC-GARCH).
#
#   [Audit Issue 2 -- VaR backtest look-ahead bug]
#     v5's `backtest_var` trained on returns.iloc[t-window:t] (information
#     through t-1) but evaluated the violation at t+1, an unintended 1-step
#     gap that invalidates the Kupiec/Christoffersen tests. FIXED: the
#     realized return is now evaluated at exactly t.
#
#   [Audit Issue 3 -- DCC-GARCH apples-to-oranges + artificial boundary]
#     v5 compared a genuine DCC-GARCH(1,1) (raw) against a rolling-window
#     OLS-residual Pearson correlation (macro-controlled) -- two different
#     estimators. FIXED: the macro-controlled comparison now fits the
#     IDENTICAL DCC-GARCH(1,1) model to the OLS-residualized (DXY/VIX
#     partialled-out) return series, so both lines in Fig. 11 use the same
#     estimator. Separately, the DCC optimizer previously hard-capped the
#     persistence parameter b at 0.95 (and a at 0.3) via `sigmoid(x)*const`;
#     the solver landed exactly on the artificial boundary. FIXED: both
#     parameters now range over (0,1) via a plain sigmoid, with only the
#     genuine stationarity constraint a+b<0.9999 enforced.
#
#   [Audit Issue 4 -- EGARCH persistence/half-life not reconciled with code]
#     v5's manuscript text reported persistence as alpha+beta and a half-life
#     of several weeks, but the BIC-selected model was frequently EGARCH,
#     whose persistence is beta alone (log-variance AR coefficient) -- and no
#     code computed a half-life at all. FIXED: `_model_persistence_halflife`
#     now computes persistence with the definition matching whichever model
#     family BIC actually selected (beta for EGARCH; alpha+beta+0.5*gamma for
#     GJR-GARCH; alpha+beta for GARCH), derives the half-life from it, prints
#     and exports it, and embeds the live numbers in the Fig. 4 caption.
#
#   [Audit Issue 5 -- hardcoded reproduced figures vs. genuine estimation]
#     v5 already distinguished "empirical" from "reproduced/conceptual" in
#     docstrings and footnotes; v6 makes this machine-checkable by adding
#     `export_figure_manifest()`, a single table classifying every figure by
#     evidentiary status and data source (menu item 19 / run automatically
#     as part of "Run all").
#
#   [Bonus fix A -- Ljung-Box divisor bug]
#     The hand-rolled Ljung-Box implementation divided the raw autocovariance
#     by (n-k) when building the ACF, then divided by (n-k) AGAIN inside the
#     Q-statistic formula, double-counting the small-sample correction and
#     inflating Q at higher lags. FIXED: prefers statsmodels' tested
#     `acorr_ljungbox`; the manual fallback now uses a single consistent ACF
#     divisor so (n-k) appears exactly once, as the textbook formula requires.
#
#   [Bonus fix B -- non-reproducible end date]
#     v5's `__main__` fetched data through `datetime.now()`, meaning the
#     "regenerates every figure and table" claim in the paper's Data
#     Availability statement was false on re-run in the future (results would
#     silently drift as new market data accrued). FIXED: the fetch window now
#     ends one day after the stated FINAL_ANALYSIS_DATE, so the script is
#     actually reproducible against the numbers reported in the paper.
#
# REQUIRED DEPENDENCIES
# ------------------------------------------------------------------------------
#   pip install yfinance arch statsmodels
# ==============================================================================

import os
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

try:
    import statsmodels  # noqa: F401
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


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

# v6: renamed/versioned cache filename. This is deliberate, not cosmetic --
# a stale v5 cache file was already built with the buggy joint-dropna() load
# path, so silently reusing it would reintroduce the calendar-truncation bug
# even after the code itself was fixed.
CACHE_FILENAME = "research_data_native_calendar_v7.csv"

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
# 2. OUTPUT HELPERS (unchanged from v5)
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
# 3. MARKET DATA LOADING -- v6, root-cause fix for Audit Issue 1
# ==============================================================================

def fetch_ticker_series(ticker, start_date, end_date):
    """Fetch ONE ticker's Close price series on its own native calendar.
    No cross-asset alignment is performed here -- this is the key structural
    change from v5: each asset keeps every date yfinance returns for it, so
    Bitcoin's weekend observations are never silently discarded by another
    asset's missing weekend data."""
    raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=ticker)
    close = raw['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.name = ticker
    return close


def load_all_series(start_date, end_date, cache_filename):
    """Returns dict[ticker] -> pd.Series, each on its OWN native calendar.

    v6: this is the root-cause fix for Audit Issue 1. v5 called
    `yf.download(all_tickers, ...)['Close'].dropna()`, which joins every
    ticker BEFORE any analysis runs -- silently intersecting Bitcoin's
    7-day-a-week calendar with equities/FX that never trade on weekends,
    while every downstream figure still annualized Bitcoin with sqrt(365).
    Here, no cross-asset intersection happens at load time at all. The only
    place it happens anywhere in this script is `build_aligned_panel`,
    called explicitly and narrowly by the two analyses that mathematically
    require paired same-day observations (rolling correlation, DCC-GARCH).
    """
    all_tickers = sorted(set(list(TICKERS.values()) + list(ASSETS_FOR_VOL_COMP.keys())))

    if os.path.exists(cache_filename):
        try:
            wide = pd.read_csv(cache_filename, index_col='Date', parse_dates=True)
            if all(t in wide.columns for t in all_tickers):
                print(f"Loading data from cache: {cache_filename} "
                      f"(each column kept on its own native calendar)...")
                return {t: wide[t].dropna() for t in all_tickers}
            print("Cache is missing required tickers. Refetching.")
        except Exception as e:
            print(f"Cache read failed ({e}). Refetching.")

    if not HAS_YFINANCE:
        raise ConnectionError("`yfinance` is not installed. Run: pip install yfinance")

    print("Fetching data from yfinance (one ticker at a time; no cross-asset alignment at load time)...")
    series = {}
    for t in all_tickers:
        print(f"  -> {t}")
        s = fetch_ticker_series(t, start_date, end_date)
        if s.empty:
            raise ValueError(f"yfinance returned no data for {t}.")
        series[t] = s

    wide = pd.concat([series[t].rename(t) for t in all_tickers], axis=1)  # outer join: nothing lost
    wide.index.name = 'Date'
    wide.to_csv(cache_filename)
    print(f"Cached to {cache_filename}")
    return series


def build_aligned_panel(series_dict, tickers):
    """The ONLY place cross-asset date intersection happens in this script.
    Use exclusively for analyses that mathematically require simultaneous
    paired observations across assets (rolling correlation, DCC-GARCH,
    OLS residualization against DXY/VIX). Restricts to the common
    trading-day calendar shared by the requested tickers -- since that
    calendar necessarily excludes weekends (equities/FX don't trade then),
    any Bitcoin annualization figure computed FROM this panel should use
    N=252, not N=365; this is stated explicitly wherever the panel is used."""
    df = pd.concat([series_dict[t].rename(t) for t in tickers], axis=1)
    return df.dropna()


def log_returns_pct(price_series):
    """Daily log returns in percent, on the series' own native calendar."""
    return np.log(price_series / price_series.shift(1)).dropna() * 100


# ==============================================================================
# 4. STATISTICAL TOOLKIT
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


def ljung_box_test(x, lags=(10, 20)):
    """Ljung-Box test for residual autocorrelation.

    v6 fix (Bonus fix A): the v5 hand-rolled implementation divided the raw
    autocovariance by (n-k) when building the ACF, then divided by (n-k)
    AGAIN inside the Q-statistic formula -- double-counting the small-sample
    correction and inflating Q (understating p-values) at higher lags
    relative to n. This prefers statsmodels' tested implementation; the
    manual fallback below uses a single, consistent ACF divisor so the
    (n-k) correction appears exactly once, in the Q formula itself, matching
    the textbook definition."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(x, lags=list(lags), return_df=True)
        return pd.DataFrame({
            "lag": lb.index.values,
            "Q_stat": lb["lb_stat"].values,
            "p_value": lb["lb_pvalue"].values,
        })
    except ImportError:
        n = len(x)
        xc = x - x.mean()
        c_full = np.correlate(xc, xc, mode='full')[n - 1:]  # raw sums, lag 0..n-1
        rho = c_full / c_full[0]                             # single consistent divisor cancels
        rows = []
        for h in lags:
            r = rho[1:h + 1]
            k = np.arange(1, h + 1)
            Q = n * (n + 2) * np.sum((r ** 2) / (n - k))
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
    """Rolling out-of-sample 1-day-ahead VaR backtest.

    v6 fix (Audit Issue 2): the training window uses information through
    t-1 (indices t-window .. t-1). The v5 code then evaluated the violation
    at index t+1 instead of t, introducing an unintended 1-step-ahead gap
    that formally invalidates the Kupiec proportion-of-failures and
    Christoffersen independence tests (they assume the realization checked
    is exactly the one the forecast targets). Fixed: `realized` is now
    `returns.iloc[t]`, matching the training window exactly."""
    returns = returns.dropna()
    n = len(returns)
    hits, idx = [], []
    for t in range(window, n):
        train = returns.iloc[t - window:t]
        var_t = -np.quantile(train, alpha)
        realized = returns.iloc[t]
        hits.append(1 if realized < -var_t else 0)
        idx.append(returns.index[t])
    hit_series = pd.Series(hits, index=idx)
    kupiec = kupiec_pof_test(hit_series, alpha)
    christoffersen = christoffersen_independence_test(hit_series)
    return hit_series, kupiec, christoffersen


def ols_residualize(y, X_df):
    """Static OLS residuals of y on X_df's columns (with intercept), aligned
    on their shared index. Returns (residual_series, beta_array). Used to
    orthogonalize Bitcoin and S&P 500 returns against the Dollar Index and
    the VIX before re-estimating DCC-GARCH on the residuals (Audit Issue 3)."""
    df = pd.concat([y.rename("_y"), X_df], axis=1).dropna()
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in X_df.columns])
    beta, *_ = np.linalg.lstsq(X, df["_y"].values, rcond=None)
    fitted = X @ beta
    resid = pd.Series(df["_y"].values - fitted, index=df.index, name=y.name)
    return resid, beta


# ==============================================================================
# 5. GARCH MODEL RACE -- v6 adds persistence/half-life reconciled with model type
# ==============================================================================

def _model_persistence_halflife(res, model_name):
    """Return (persistence, half_life_days, formula_str), using the
    definition that matches the selected model family.

    v6 fix (Audit Issue 4): v5's manuscript text reported persistence as
    alpha+beta and a half-life of "several weeks to two months" regardless
    of which model BIC actually selected -- and no code computed a half-life
    at all, so the editor could not trace the figure. In Nelson's EGARCH,
    the conditional variance is modeled in LOGS, and persistence is
    determined solely by the autoregressive coefficient beta, not
    alpha+beta; the alpha term there governs the sign/magnitude response to
    shocks, not variance persistence in the GARCH sense. GJR-GARCH adds an
    asymmetry term gamma, whose expected contribution to persistence is
    0.5*gamma under a symmetric innovation density."""
    p = res.params
    alpha = p.get("alpha[1]", np.nan)
    beta = p.get("beta[1]", np.nan)
    gamma = p.get("gamma[1]", np.nan)

    if model_name == "EGARCH":
        persistence = beta
        formula = "beta (EGARCH log-variance AR coefficient)"
    elif model_name == "GJR-GARCH":
        g = 0.0 if (gamma is None or (isinstance(gamma, float) and np.isnan(gamma))) else gamma
        persistence = alpha + beta + 0.5 * g
        formula = "alpha + beta + 0.5*gamma (GJR-GARCH, symmetric-innovation persistence)"
    else:
        persistence = alpha + beta
        formula = "alpha + beta (GARCH)"

    if pd.notna(persistence) and 0 < persistence < 1:
        half_life = np.log(0.5) / np.log(persistence)
    else:
        half_life = np.nan
    return persistence, half_life, formula


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

    lb_levels = ljung_box_test(std_resid.values, lags=(10, 20))
    lb_squares = ljung_box_test((std_resid.values) ** 2, lags=(10, 20))

    persistence, half_life, pers_formula = _model_persistence_halflife(best_res, best_model)

    print(f"\n--- GARCH Model Race: {asset_name} ---")
    print(comp_df.to_string(index=False))
    print(f"\nSelected by BIC: {best_model} ({best_dist})")
    if pd.notna(persistence):
        print(f"Persistence [{pers_formula}]: {persistence:.4f}")
    else:
        print("Persistence: undefined (required parameter missing from fit).")
    if pd.notna(half_life):
        print(f"Implied volatility-shock half-life: {half_life:.1f} days (~{half_life / 7:.1f} weeks)")
    else:
        print("Implied half-life: undefined (persistence outside (0,1)).")
    print("\nLjung-Box on standardized residuals:")
    print(lb_levels.to_string(index=False))
    print("\nLjung-Box on SQUARED standardized residuals:")
    print(lb_squares.to_string(index=False))

    export_table(comp_df, f"table_garch_race_{asset_name.lower().replace(' ', '_')}",
                 caption=f"GARCH-family model comparison for {asset_name} log returns (ranked by BIC).",
                 label=f"tab:garch_{asset_name.lower()}")

    pers_df = pd.DataFrame([{
        "Asset": asset_name, "Model": best_model, "Distribution": best_dist,
        "Persistence": persistence, "Persistence_Formula": pers_formula,
        "Half_Life_Days": half_life,
    }])
    export_table(pers_df, f"table_persistence_halflife_{asset_name.lower().replace(' ', '_')}",
                 caption=f"Volatility persistence and implied shock half-life for {asset_name}, "
                         f"using the definition matching the BIC-selected model family.",
                 label=f"tab:persistence_{asset_name.lower()}")

    diagnostics = {"levels": lb_levels, "squares": lb_squares,
                    "model": best_model, "dist": best_dist,
                    "persistence": persistence, "half_life_days": half_life,
                    "persistence_formula": pers_formula}
    return best_res, comp_df, diagnostics


# ==============================================================================
# 6. DCC-GARCH -- v6 fixes the boundary bug and the estimator mismatch
# ==============================================================================

def _logit(p):
    return np.log(p / (1 - p))


def _fit_dcc_core(Z, max_iter=400, x0=None):
    """Two-step DCC(1,1) correlation-dynamics optimizer (Engle, 2002),
    operating directly on a T x N matrix of GARCH-standardized residuals Z.

    Deliberately separated from the univariate-GARCH standardization step
    (which requires the `arch` package) so this core optimization can be
    unit-tested and reused independently of network/data access.

    v6 fix (Audit Issue 3, boundary bug): v5 capped b at a hard 0.95 ceiling
    (and a at 0.3) via `sigmoid(x) * const`; the solver landed exactly on
    the artificial 0.950 boundary. Those per-parameter caps are not an
    econometric requirement -- the only genuine constraint on a DCC(1,1)
    process is a>=0, b>=0, a+b<1. Here both parameters map to (0,1) via a
    plain sigmoid, and only the joint stationarity constraint a+b<0.9999 is
    enforced, via proportional rescaling when violated."""
    Z = np.asarray(Z, dtype=float)
    T, N = Z.shape
    Qbar = np.atleast_2d(np.cov(Z.T))

    def unpack(theta):
        ra, rb = theta
        a = 1.0 / (1.0 + np.exp(-ra))
        b = 1.0 / (1.0 + np.exp(-rb))
        s = a + b
        if s >= 0.9999:
            scale = 0.9999 / s
            a, b = a * scale, b * scale
        return a, b

    def neg_loglik(theta):
        a, b = unpack(theta)
        Qt = Qbar.copy()
        ll = 0.0
        for t in range(T):
            if t > 0:
                zt1 = Z[t - 1].reshape(-1, 1)
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
            zt = Z[t].reshape(-1, 1)
            ll += 0.5 * (logdet + (zt.T @ Rinv @ zt).item())
        return ll

    if x0 is None:
        # A reasonable DCC starting point (small news-impact a, high
        # persistence b, typical of financial-return correlations) rather
        # than an arbitrary guess that happened to sit near the old
        # artificial boundary.
        x0 = np.array([_logit(0.03), _logit(0.93)])

    opt = minimize(neg_loglik, x0=x0, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5})
    a, b = unpack(opt.x)

    Qt = Qbar.copy()
    corr_series = np.full(T, np.nan)
    for t in range(T):
        if t > 0:
            zt1 = Z[t - 1].reshape(-1, 1)
            Qt = (1 - a - b) * Qbar + a * (zt1 @ zt1.T) + b * Qt
        d = np.sqrt(np.diag(Qt))
        Rt = Qt / np.outer(d, d)
        if N >= 2:
            corr_series[t] = Rt[0, 1]

    return corr_series, {"a": a, "b": b, "persistence": a + b, "converged": bool(opt.success)}


def dcc_garch_bivariate(returns_df, asset_names, max_iter=400):
    """Two-step DCC-GARCH(1,1) (Engle, 2002): fit a univariate GARCH(1,1)-t
    model to each series, standardize, then fit the DCC correlation
    dynamics via `_fit_dcc_core`.

    v6: this SAME function is now called on both the raw return series and
    the DXY/VIX-residualized return series (see fig10_11), so the "raw" and
    "macro-controlled" lines in Fig. 11 are estimated with the identical
    econometric method -- fixing the v5 DCC-vs-rolling-Pearson estimator
    mismatch (Audit Issue 3)."""
    if not HAS_ARCH:
        print("  [ERROR] `arch` is not installed. Run: pip install arch")
        return None, None

    std_resid = {}
    for col in asset_names:
        am = arch_model(returns_df[col].dropna(), vol='Garch', p=1, q=1, dist='t')
        res = am.fit(disp='off')
        std_resid[col] = res.resid / res.conditional_volatility
    Z_df = pd.concat(std_resid, axis=1).dropna()
    Z_df.columns = asset_names

    corr_series, params = _fit_dcc_core(Z_df.values, max_iter=max_iter)
    if not params["converged"]:
        print(f"  [WARN] DCC optimizer did not report convergence for {asset_names}; "
              f"treat a={params['a']:.4f}, b={params['b']:.4f} as approximate.")
    dcc = pd.Series(corr_series, index=Z_df.index, name=f"DCC_{asset_names[0]}_{asset_names[1]}")
    print(f"DCC-GARCH(1,1) estimated [{', '.join(asset_names)}]: "
          f"a={params['a']:.4f}, b={params['b']:.4f}, persistence(a+b)={params['persistence']:.4f}")
    return dcc, params


def rolling_partial_correlation(returns_df, x_col, y_col, control_cols, window=60):
    """Rolling PARTIAL Pearson correlation between x_col and y_col,
    controlling linearly for control_cols within each window.

    NOTE (v6): this function is kept as an optional supplementary
    robustness diagnostic only. It is intentionally NOT used for the main
    Fig. 11 "raw vs. macro-controlled" comparison, because comparing it
    directly to a DCC-GARCH line would reproduce the exact apples-to-oranges
    estimator mismatch that Audit Issue 3 flagged (a rolling, equal-weighted
    Pearson correlation is not the same class of estimator as a dynamic
    conditional correlation). Fig. 11 instead compares two DCC-GARCH fits
    to each other (see `dcc_garch_bivariate`)."""
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

# ==============================================================================
# 7. FIGURE FUNCTIONS -- MAIN TEXT (Fig. 1: framework; Figs. 2-6: Pillars I-III)
# ==============================================================================

def fig01_reintermediation_trilemma():
    """
    NEW (v7): the paper's central conceptual figure. Not derived from data --
    a diagrammatic summary of the argument developed empirically in Sections
    3-5: the same class of economic force (scale economies in liquidity
    provision and coordination costs) reproduces custodial intermediation at
    the settlement, routing, and validation layers independently, despite
    each layer's protocol-level design for disintermediation."""
    print("\nGenerating Figure 1: The Re-Intermediation Trilemma (conceptual synthesis diagram)...")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')

    col_w = 3.2
    xs = [0.6, 4.4, 8.2]
    layers = [
        {"title": "Settlement (L1)", "color": CB["blue"],
         "lines": ["Ideal: peer-to-peer\nfinal settlement",
                   "Force: throughput/finality\nceiling vs. retail scale",
                   "Outcome: RTGS-scale only\n(\u00a73, Fig. 2)"]},
        {"title": "Routing (Lightning)", "color": CB["vermillion"],
         "lines": ["Ideal: decentralized\nmesh routing",
                   "Force: channel-capital\nopportunity cost",
                   "Outcome: custodial hubs\n(\u00a74, Figs. 3-4)"]},
        {"title": "Validation (Mining)", "color": CB["green"],
         "lines": ["Ideal: dispersed\nhashpower",
                   "Force: super-additive\ncoalition value",
                   "Outcome: pool concentration\n(\u00a75, Figs. 5-6)"]},
    ]

    header_y0, header_h = 7.0, 1.3
    sub_y0s = [5.55, 4.05, 2.55]
    sub_h = 1.2

    for x, layer in zip(xs, layers):
        ax.add_patch(mpatches.FancyBboxPatch((x, header_y0), col_w, header_h,
                     boxstyle="round,pad=0.08", facecolor=layer["color"], alpha=0.92,
                     edgecolor='black'))
        ax.text(x + col_w / 2, header_y0 + header_h / 2, layer["title"], ha='center', va='center',
                fontsize=12.5, weight='bold', color='white')
        for sub_y0, line in zip(sub_y0s, layer["lines"]):
            ax.add_patch(mpatches.FancyBboxPatch((x, sub_y0), col_w, sub_h,
                         boxstyle="round,pad=0.06", facecolor='white',
                         edgecolor=layer["color"], lw=1.6))
            ax.text(x + col_w / 2, sub_y0 + sub_h / 2, line, ha='center', va='center', fontsize=8.8)
        ax.annotate('', xy=(x + col_w / 2, 1.95), xytext=(x + col_w / 2, sub_y0s[-1] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color=layer["color"]))

    ax.add_patch(mpatches.FancyBboxPatch((1.4, 0.3), 9.2, 1.55, boxstyle="round,pad=0.08",
                 facecolor=CB["grey"], alpha=0.94, edgecolor='black'))
    ax.text(6.0, 1.075,
            "Same mechanism at every layer: economies of scale in liquidity provision\n"
            "and coordination costs reassert custodial intermediation\n"
            "(Diamond, 1984; Katz & Shapiro, 1985)",
            ha='center', va='center', fontsize=9.5, weight='bold', color='white')

    ax.set_title("The Re-Intermediation Trilemma: Three Layers, One Mechanism (Fig. 1)",
                 fontsize=13.5)
    footnote(fig, "Conceptual synthesis diagram, not derived from data; summarizes the argument "
                   "developed empirically in Sections 3-5.")
    plt.tight_layout()
    save_fig(fig, "figure_01_reintermediation_trilemma")
    plt.close(fig)


def fig02_settlement_layer_comparison():
    """
    Pillar I (Sec. 3). Fedwire TPS uses the Fed's own 2024 PFMI disclosure
    (836,322 avg daily transactions -> ~9.68 TPS). T2 is shown as a RANGE
    (400k-450k payments/day) since the ECB's TARGET Services Annual Report
    2024 does not publish a single clean annual transaction count."""
    print("\nGenerating Figure 2: Settlement-Layer-Matched Throughput Comparison (Pillar I)...")
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

    fig.suptitle('Transaction Throughput by Settlement Layer (Fig. 2)', fontsize=15)
    footnote(fig, "Fedwire TPS uses the Fed's own 2024 PFMI disclosure (836,322 avg. daily "
                   "transactions). T2's exact 2024 transaction count is not cleanly published; "
                   "shown here as an order-of-magnitude RANGE with that uncertainty disclosed, not "
                   "a false-precision point estimate. This figure supports the Pillar I argument "
                   "that Bitcoin L1 is throughput-adequate for a settlement layer and "
                   "throughput-inadequate for a retail network -- not that either comparison alone "
                   "is dispositive (\u00a73).")
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    save_fig(fig, "figure_02_settlement_layer_comparison")
    plt.close(fig)


def fig03_ln_topology_schematic():
    """Pillar II (Sec. 4), theory panel: the game-theoretic prediction, shown
    before the empirical test (Fig. 4)."""
    print("\nGenerating Figure 3: LN Topology, Mesh vs. Hub-and-Spoke (Pillar II, theory)...")
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
    fig.suptitle('Predicted Topological Evolution of the Lightning Network (Fig. 3)\n'
                 'CONCEPTUAL SCHEMATIC, not a simulation or measurement', fontsize=12)
    footnote(fig, "Deterministic, hand-placed schematic illustrating the mesh-to-hub-and-spoke "
                   "prediction formalized by Avarikioti et al. (2020); tested empirically in Fig. 4.")
    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    save_fig(fig, "figure_03_ln_topology_schematic")
    plt.close(fig)


def fig04_ln_reliability_synthesis():
    """Pillar II (Sec. 4), evidence panel. Three independent bodies of
    evidence, not one study set against one company's self-report:
    (1) the 2020 mesh-wide reliability probe (Waugh & Holz); (2) a single
    well-capitalized custodial hub's own 2023 platform success rate (River
    Financial); (3) the structural evidence -- Gini coefficient of channel-
    capacity concentration computed independently across several peer-
    reviewed studies (Lin et al. 2020; Zabka et al. 2022; Atmanaviciute
    et al. 2025), continuously over 2018-2025. Panel 3 is what upgrades this
    figure from an anecdote-vs-anecdote comparison to a decade-spanning,
    methodologically independent, convergent empirical record."""
    print("\nGenerating Figure 4: LN Reliability and Structural-Concentration Synthesis "
          "(Pillar II, evidence)...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.5))

    amounts = ['$0.01', '$10', '$50']
    success = [72, 44.15, 30.93]
    bars = ax1.bar(amounts, success, color=[CB["green"], CB["orange"], CB["vermillion"]])
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Payment Amount')
    ax1.set_ylabel('Routing Success Rate (%)')
    ax1.set_title('2020: Mesh-Wide Active Probe\n(Waugh & Holz, 4,626 nodes)', fontsize=10.5)
    for b, v in zip(bars, success):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:g}%', ha='center', fontsize=9.5, weight='bold')

    labels2 = ['River platform\n(Aug 2023, 308k txns)']
    vals2 = [99.7]
    bars2 = ax2.bar(labels2, vals2, color=CB["blue"], width=0.5)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Payment Success Rate (%)')
    ax2.set_title('2023: Single Well-Capitalized\nCustodial Hub', fontsize=10.5)
    for b, v in zip(bars2, vals2):
        ax2.text(b.get_x() + b.get_width()/2, v + 1.5, f'{v:g}%', ha='center', fontsize=10.5, weight='bold')

    years3 = ['2018', '2023', '2025']
    gini3 = [0.86, 0.955, 0.97]
    bars3 = ax3.bar(years3, gini3, color=[CB["sky"], CB["orange"], CB["vermillion"]])
    ax3.set_ylim(0, 1.05)
    ax3.axhline(1.0, color='black', lw=0.6, ls=':')
    ax3.set_ylabel('Gini Coefficient, Channel-Capacity Distribution')
    ax3.set_title('2018-2025: Structural Concentration\n(independent studies, converging)', fontsize=10.5)
    for b, v in zip(bars3, gini3):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9.5, weight='bold')

    fig.suptitle('Lightning Network: Reliability and Structural-Concentration Synthesis (Fig. 4)',
                 fontsize=14)
    footnote(fig, "Panels 1-2 are NOT like-for-like measurements of the same quantity: panel 1 is "
                   "decentralized mesh-wide reachability, panel 2 is one large custodial node's own "
                   "platform success rate. Panel 3 is structurally different evidence again -- an "
                   "inequality metric on the public channel graph, computed independently by "
                   "separate research teams using separate methodologies (Lin, Primicerio, "
                   "Squartini, Decker & Tessone, 2020, New J. Phys.; Zabka, Foerster, Decker & "
                   "Schmid, 2022, FC; Atmanaviciute, Vanagas & Masteika, 2025, IEEE Access; exact "
                   "point estimates differ slightly by study and methodology, so values shown here "
                   "are representative anchors within each study's reported range, not a single "
                   "author's precise series) -- and it moves in the same direction as panels 1-2 "
                   "continuously across seven years, not at one measurement date. This convergence "
                   "across independent methods is the strongest evidence in this paper for the "
                   "hub-formation prediction of Fig. 3, considerably stronger than panels 1-2 alone.")
    plt.tight_layout(rect=[0, 0.09, 1, 0.90])
    save_fig(fig, "figure_04_ln_reliability_synthesis")
    plt.close(fig)


def fig05_security_budget_dilemma():
    """Pillar III (Sec. 5), theory panel: Budish's (2018) flow-vs-stock
    security condition under two Layer-2-adoption scenarios."""
    print("\nGenerating Figure 5: Security Budget Dilemma (Pillar III, theory)...")
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
    ax.set_title("A Model of the Bitcoin Security-Budget Dilemma (Fig. 5)\nTHEORETICAL MODEL", fontsize=12)
    ax.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig, "figure_05_security_budget_dilemma")
    plt.close(fig)


def fig06_mining_pool_concentration():
    """Pillar III (Sec. 5), evidence panel: hashrate concentration as a
    range across independent trackers, plus the Stratum V2 caveat (mitigates
    pool-level censorship specifically, not concentration itself)."""
    print("\nGenerating Figure 6: Mining Pool Concentration (Pillar III, evidence)...")
    labels = ['Top 2 Pools\n(Foundry USA + AntPool)', 'Top 4-5 Pools\n(cumulative)']
    low = [43, 65]
    high = [57, 76]
    mid = [(l+h)/2 for l, h in zip(low, high)]
    err = [(h-l)/2 for l, h in zip(low, high)]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    bars = ax.barh(labels, mid, xerr=err, capsize=8, color=[CB["vermillion"], CB["orange"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Cumulative Share of Network Hashrate (%), range across trackers')
    ax.set_title('Bitcoin Mining Pool Concentration, 2025-2026 (Fig. 6)')
    for b, l, h in zip(bars, low, high):
        ax.text(h + 2, b.get_y() + b.get_height()/2, f'{l}-{h}%', va='center', fontsize=10, weight='bold')
    ax.text(2, -1.15, 'Stratum V2: ~75% of hashrate joined the working group (May 2026);\n'
                       'only ~3-5% runs it in production. Mitigates pool-level censorship\n'
                       'specifically -- does NOT reduce hashrate concentration or reorg risk.',
                       fontsize=8, style='italic', color='#444444', transform=ax.transData)
    footnote(fig, "REPRODUCED AS A RANGE across independent trackers (Hashrate Index; B10C, 2025; "
                   "Spark, 2026), since single-point estimates for this metric swing by several "
                   "points week to week and no single tracker should be presented as definitive. "
                   "The same super-additive-coalition-value logic that predicts Lightning hub "
                   "formation (Fig. 3) predicts this concentration (Leonardos et al., 2019) -- "
                   "\u00a75.")
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    save_fig(fig, "figure_06_mining_pool_concentration")
    plt.close(fig)


# ==============================================================================
# 7b. FIGURE FUNCTIONS -- APPENDIX A (volatility, risk, correlation: context,
#     not part of the re-intermediation argument)
# ==============================================================================

def figA1_rolling_volatility(series_dict):
    print("\nGenerating Figure A1: Comparative Rolling Volatility (Appendix A -- context only)...")
    windows = [15, 200]
    colors = {'Apple': CB["sky"], 'Bitcoin': CB["orange"], 'Gold': CB["vermillion"]}
    vol_data = {}
    for ticker, name in ASSETS_FOR_VOL_COMP.items():
        if ticker not in series_dict:
            print(f"  [WARN] {ticker} ({name}) not available; skipping.")
            continue
        s = series_dict[ticker].dropna()
        ret = s.pct_change().dropna()
        ann = np.sqrt(TRADING_DAYS.get(name, 252))
        for w in windows:
            vol_data[(name, w)] = ret.rolling(w).std() * ann

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9), sharex=True)
    for w, ax, label in zip(windows, axes, ["Short-Term (15-Day)", "Long-Term (200-Day)"]):
        for name in ASSETS_FOR_VOL_COMP.values():
            s = vol_data.get((name, w))
            if s is not None:
                s.plot(ax=ax, color=colors[name], lw=1.8, label=name)
        ax.set_ylabel(f"{label} Ann. Volatility")
        ax.legend(loc="upper left")
    fig.suptitle("Comparative Rolling Volatility: Bitcoin, Gold, Apple (Fig. A1)", fontsize=15)
    footnote(fig, "Each asset's rolling-volatility window and annualization factor use its OWN "
                   "native calendar -- 365 calendar days for Bitcoin (which trades every day), 252 "
                   "trading days for equities/gold -- computed independently, with no prior "
                   "cross-asset date intersection. Supplementary context; not part of the "
                   "re-intermediation argument developed in the main text.")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, "figure_A1_rolling_volatility")
    plt.close(fig)
    return vol_data


def figA2_A3_risk_suite(series_dict):
    print("\nGenerating Figures A2-A3: VaR/ES Suite, Backtests, and GARCH Model Race "
          "(Appendix A -- context only)...")

    rows = []
    returns_by_asset = {}
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        ticker = TICKERS[name]
        if ticker not in series_dict:
            continue
        r = log_returns_pct(series_dict[ticker])
        returns_by_asset[name] = r
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
    export_table(var_df, "table_A_var_es_comparison",
                 caption="1-day 95% VaR and Expected Shortfall by method (percent). Each asset's "
                         "returns use its own native trading calendar.", label="tab:var_es")

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(var_df))
    width = 0.25
    ax.bar(x - width, var_df["Historical_VaR95"], width, label="Historical VaR", color=CB["blue"])
    ax.bar(x, var_df["CornishFisher_VaR95"], width, label="Cornish-Fisher VaR", color=CB["orange"])
    ax.bar(x + width, var_df["EVT_VaR95"], width, label="EVT (POT/GPD) VaR", color=CB["vermillion"])
    ax.set_xticks(x); ax.set_xticklabels(var_df["Asset"])
    ax.set_ylabel("1-Day 95% VaR (%)")
    ax.set_title("1-Day 95% VaR by Method (Fig. A2)")
    ax.legend()
    footnote(fig, "EVT VaR uses a Peaks-Over-Threshold GPD fit (McNeil & Frey, 2000); "
                   "Cornish-Fisher adjusts for skewness/kurtosis. Each asset's returns are computed "
                   "on its own native calendar. Supplementary context; not part of the "
                   "re-intermediation argument developed in the main text.")
    plt.tight_layout()
    save_fig(fig, "figure_A2_var_comparison")
    plt.close(fig)

    bt_rows = []
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        r = returns_by_asset.get(name)
        if r is None or len(r) < 300:
            continue
        hit_series, kupiec, christoffersen = backtest_var(r, alpha=0.05, window=250)
        bt_rows.append({
            "Asset": name, "N_obs": kupiec["n"], "Violations": kupiec["violations"],
            "Expected": kupiec["expected"], "Kupiec_LR": kupiec["LR_stat"],
            "Kupiec_p": kupiec["p_value"], "Christoffersen_LR": christoffersen["LR_ind"],
            "Christoffersen_p": christoffersen["p_value"],
        })
    bt_df = pd.DataFrame(bt_rows)
    print("\nRolling out-of-sample VaR backtests (strict 1-step-ahead; train ends at t-1, "
          "evaluated at t):")
    print(bt_df.to_string(index=False))
    export_table(bt_df, "table_A_var_backtests",
                 caption="Kupiec and Christoffersen VaR backtests, strict 1-step-ahead evaluation.",
                 label="tab:var_backtest")

    btc_returns = returns_by_asset.get('Bitcoin')
    if btc_returns is not None:
        btc_res, garch_comp, garch_diag = run_garch_model_race(btc_returns, "Bitcoin")
    else:
        btc_res, garch_comp, garch_diag = None, None, None

    if btc_res is not None:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(btc_returns.index, btc_returns, color=CB["grey"], alpha=0.6,
                 lw=0.8, label='Daily Log Return (%)')
        ax2.plot(btc_res.conditional_volatility.index, btc_res.conditional_volatility,
                 color=CB["vermillion"], lw=1.6,
                 label=f'Conditional Volatility ({garch_diag["model"]}-{garch_diag["dist"]}, BIC-selected)')
        ax2.set_title('Bitcoin Returns and BIC-Selected Conditional Volatility (Fig. A3)')
        ax2.set_ylabel('Percent (%)')
        ax2.legend()
        hl = garch_diag["half_life_days"]
        hl_txt = (f", implied half-life ~{hl:.1f} days (~{hl / 7:.1f} weeks)"
                   if pd.notna(hl) else ", half-life undefined for the selected fit")
        footnote(fig2, f"Model selected from a 9-specification race by BIC: "
                        f"{garch_diag['model']}-{garch_diag['dist']}. Persistence "
                        f"[{garch_diag['persistence_formula']}] = {garch_diag['persistence']:.3f}"
                        f"{hl_txt}. Persistence and half-life use the definition matching the "
                        f"selected model family (see table_A_persistence_halflife_bitcoin.csv). "
                        f"Supplementary context; not part of the re-intermediation argument.")
        plt.tight_layout()
        save_fig(fig2, "figure_A3_garch_volatility")
        plt.close(fig2)

        pers_df = pd.DataFrame([{
            "Asset": "Bitcoin", "Model": garch_diag["model"], "Distribution": garch_diag["dist"],
            "Persistence": garch_diag["persistence"], "Persistence_Formula": garch_diag["persistence_formula"],
            "Half_Life_Days": garch_diag["half_life_days"],
        }])
        export_table(pers_df, "table_A_persistence_halflife_bitcoin",
                     caption="Volatility persistence and implied shock half-life for Bitcoin, "
                             "using the definition matching the BIC-selected model family.",
                     label="tab:persistence_bitcoin")

    return var_df, bt_df, garch_comp


def figA4_A5_digital_gold_robustness(series_dict):
    print("\nGenerating Figure A4: Drawdowns (Appendix A -- context only)...")
    btc = series_dict[TICKERS['Bitcoin']].dropna()
    dd = (btc - btc.cummax()) / btc.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax1.plot(btc.index, btc, color=CB["blue"], label='Bitcoin Price (USD)')
    ax1.set_yscale('log'); ax1.legend(); ax1.set_ylabel('Price (log scale)')
    ax1.set_title('Bitcoin Price and Historical Drawdowns (Fig. A4)')
    ax2.plot(dd.index, dd*100, color=CB["vermillion"])
    ax2.fill_between(dd.index, dd*100, 0, color=CB["vermillion"], alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    max_dd = dd.min()*100
    ax2.text(dd.idxmin(), max_dd, f'Max DD: {max_dd:.1f}%', ha='right', va='top', fontsize=9)
    plt.tight_layout()
    save_fig(fig, "figure_A4_drawdowns")
    plt.close(fig)

    print("\nGenerating Figure A5: Raw vs. Macro-Liquidity-Controlled DCC-GARCH "
          "(Appendix A -- context only)...")
    pair_cols = [TICKERS['Bitcoin'], TICKERS['S&P 500']]
    aligned_pair_prices = build_aligned_panel(series_dict, pair_cols)
    log_returns_pair = np.log(aligned_pair_prices / aligned_pair_prices.shift(1)).dropna() * 100
    rolling_corr = log_returns_pair[TICKERS['Bitcoin']].rolling(60).corr(log_returns_pair[TICKERS['S&P 500']])

    fig2, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(rolling_corr.index, rolling_corr, color=CB["grey"], alpha=0.5, lw=1.0,
            label='60-Day Rolling Pearson (naive descriptive baseline)')

    if HAS_ARCH:
        dcc_raw, params_raw = dcc_garch_bivariate(log_returns_pair, pair_cols)
        if dcc_raw is not None:
            ax.plot(dcc_raw.index, dcc_raw, color=CB["vermillion"], lw=1.6,
                    label=f'DCC-GARCH(1,1), raw (a={params_raw["a"]:.3f}, b={params_raw["b"]:.3f})')

        macro_cols = [TICKERS['Bitcoin'], TICKERS['S&P 500'], TICKERS['US Dollar'], TICKERS['VIX']]
        if all(c in series_dict for c in macro_cols):
            aligned_macro_prices = build_aligned_panel(series_dict, macro_cols)
            macro_returns = np.log(aligned_macro_prices / aligned_macro_prices.shift(1)).dropna() * 100
            btc_resid, _ = ols_residualize(macro_returns[TICKERS['Bitcoin']],
                                            macro_returns[[TICKERS['US Dollar'], TICKERS['VIX']]])
            spx_resid, _ = ols_residualize(macro_returns[TICKERS['S&P 500']],
                                            macro_returns[[TICKERS['US Dollar'], TICKERS['VIX']]])
            resid_df = pd.concat([btc_resid, spx_resid], axis=1)
            resid_df.columns = pair_cols
            dcc_ctrl, params_ctrl = dcc_garch_bivariate(resid_df, pair_cols)
            if dcc_ctrl is not None:
                ax.plot(dcc_ctrl.index, dcc_ctrl, color=CB["blue"], lw=1.6, ls='--',
                        label=f'DCC-GARCH(1,1), residualized on DXY & VIX '
                              f'(a={params_ctrl["a"]:.3f}, b={params_ctrl["b"]:.3f})')
        else:
            print("  [WARN] VIX or Dollar Index data unavailable; residualized-DCC panel skipped.")
    else:
        print("  [WARN] `arch` not installed; DCC-GARCH panels skipped.")

    ax.axhline(0, color='black', linestyle='--', lw=1)
    ax.set_title('Bitcoin vs. S&P 500: Raw vs. Macro-Liquidity-Controlled DCC-GARCH (Fig. A5)')
    ax.set_ylabel('Correlation')
    ax.legend(fontsize=9)
    footnote(fig2, "The dashed blue line applies the IDENTICAL DCC-GARCH(1,1) estimator as the "
                    "solid red line, but to Bitcoin and S&P 500 returns each first residualized "
                    "(static OLS) against the Dollar Index (UUP) and the VIX. Both lines therefore "
                    "use the same econometric estimator; only the input series differ. This panel "
                    "is restricted to the common TradFi trading-day calendar, disclosed here since "
                    "Bitcoin's other statistics in this appendix use its native 365-day calendar. "
                    "Supplementary context; not part of the re-intermediation argument.")
    plt.tight_layout()
    save_fig(fig2, "figure_A5_correlation_dcc")
    plt.close(fig2)


def tableA_adf_stationarity_report(series_dict):
    print("\nGenerating Appendix A stationarity table (ADF tests, context only)...")
    rows = []
    for name in ['Bitcoin', 'US Dollar', 'Gold', 'S&P 500']:
        ticker = TICKERS[name]
        if ticker not in series_dict:
            continue
        price = series_dict[ticker].dropna()
        ret = np.log(price / price.shift(1)).dropna()
        rows.append(adf_test(price, f"{name} (level)"))
        rows.append(adf_test(ret, f"{name} (log return)"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    export_table(df, "table_A_adf_stationarity",
                 caption="Augmented Dickey-Fuller stationarity tests on price levels and log returns.",
                 label="tab:adf")
    return df


# ==============================================================================
# 7c. FIGURE FUNCTIONS -- APPENDIX B (market microstructure / liquidity
#     robustness, per the audit's suggestion to relocate rather than cut)
# ==============================================================================

def figB1_wash_trading():
    """Bitwise (2019)'s 95% estimate alongside Cong et al. (2022, NBER)'s
    more rigorous, more recent re-estimate and Sila et al. (2025); flags
    what none of these numbers describe: the now-larger regulated-venue
    (ETF/CME/MiCA) share of the market since January 2024."""
    print("\nGenerating Figure B1: Wash-Trading Estimates (Appendix B)...")
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
    ax.set_title('Wash-Trading Estimates on Unregulated Exchanges,\nThree Independent Studies (Fig. B1)')
    ax.set_ylim(0, 105)
    for b, l, h in zip(bars, low, high):
        label = f'{l:g}%' if l == h else f'{l:g}-{h:g}%'
        ax.text(b.get_x() + b.get_width()/2, h + 2, label, ha='center', fontsize=10, weight='bold')
    footnote(fig, "All three estimates describe UNREGULATED exchanges only. Since Jan. 2024, "
                   "U.S. spot Bitcoin ETF approval and MiCA implementation have shifted a "
                   "substantial share of institutional PRICE-FORMATION volume onto regulated venues "
                   "not characterized by any of these three studies. Adjacent to, but not part of, "
                   "the re-intermediation argument developed in the main text.")
    plt.tight_layout()
    save_fig(fig, "figure_B1_wash_trading_updated")
    plt.close(fig)


def figB2_tether_dominance():
    print("\nGenerating Figure B2: Tether Dominance of Stablecoin Market Cap (Appendix B)...")
    labels = ['Tether (USDT)', 'All Other Stablecoins']
    low, high = 58.3, 63.0
    vals = [(low + high) / 2, 100 - (low + high) / 2]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, vals, color=[CB["vermillion"], CB["grey"]])
    ax.set_ylabel('Share of Total Stablecoin Market Capitalization (%)')
    ax.set_title(f'Tether (USDT) Share of Stablecoin Market Cap (Fig. B2)\n(range {low:g}-{high:g}%, late 2025-2026)')
    ax.set_ylim(0, 70)
    ax.text(0, vals[0] + 1.5, f'~{low:g}-{high:g}%', ha='center', fontsize=11, weight='bold')
    footnote(fig, "Shown as a range across independent trackers (CoinMarketCap, DeFiLlama-based "
                   "aggregators) rather than a single archived figure. The GENIUS Act (enacted "
                   "July 2025) establishes forward-looking reserve/redemption standards for payment "
                   "stablecoin issuers but does not retroactively resolve concentration risk in "
                   "Tether's existing market share. Adjacent to, but not part of, the "
                   "re-intermediation argument developed in the main text.")
    plt.tight_layout()
    save_fig(fig, "figure_B2_tether_dominance")
    plt.close(fig)


# ==============================================================================
# 7d. FIGURE FUNCTIONS -- APPENDIX C (attack-cost economics: supplementary
#     detail for Pillar III, not part of the core structural claim)
# ==============================================================================

def figC1_attack_cost_breakdown():
    """Harvey's original Oct-2025 static ~$6B estimate alongside Harvey's own
    July-2026 update to ~$8B, which adds a derivatives-shorting profit
    mechanism."""
    print("\nGenerating Figure C1: 51% Attack Cost, 2025 vs. 2026 Estimates (Appendix C)...")
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

    fig.suptitle("51% Attack Economics: Harvey's Own Revision (Fig. C1)", fontsize=14)
    footnote(fig, "REPRODUCED FROM CITED SOURCES: Harvey (Oct. 2025) estimated ~$6B in static "
                   "hardware/infrastructure cost for a one-week attack. Harvey's own July 2026 "
                   "update raises this to ~$8B, adding a derivatives-shorting profit mechanism. "
                   "Both estimates should be read as bounds on a moving target. Supplementary "
                   "detail for the Pillar III security-budget argument (\u00a75); the core "
                   "structural claim does not depend on the specific dollar figure.")
    plt.tight_layout(rect=[0, 0.08, 1, 0.90])
    save_fig(fig, "figure_C1_attack_cost_breakdown")
    plt.close(fig)


# ==============================================================================
# 7e. FIGURE MANIFEST
# ==============================================================================

FIGURE_MANIFEST = [
    ("figure_01_reintermediation_trilemma", "conceptual", "Main text -- Framework (\u00a72)",
     "None -- synthesis diagram, no data"),
    ("figure_02_settlement_layer_comparison", "reproduced", "Main text -- Pillar I (\u00a73)",
     "Fed 2024 PFMI disclosure; ECB TARGET Services Annual Report 2024; Mastercard/Visa 10-K (cited)"),
    ("figure_03_ln_topology_schematic", "conceptual", "Main text -- Pillar II (\u00a74)",
     "None -- fixed-node schematic, no data"),
    ("figure_04_ln_reliability_synthesis", "reproduced", "Main text -- Pillar II (\u00a74)",
     "Waugh & Holz (2020); River Financial (2023) (cited)"),
    ("figure_05_security_budget_dilemma", "theoretical-model", "Main text -- Pillar III (\u00a75)",
     "Stylized scenario paths, no market data"),
    ("figure_06_mining_pool_concentration", "reproduced", "Main text -- Pillar III (\u00a75)",
     "Hashrate Index; B10C (2025); Spark (2026) (cited)"),
    ("figure_A1_rolling_volatility", "empirical", "Appendix A",
     "yfinance (BTC-USD, GC=F, AAPL), each on its native calendar"),
    ("figure_A2_var_comparison", "empirical", "Appendix A", "yfinance, each asset's own native calendar"),
    ("figure_A3_garch_volatility", "empirical", "Appendix A",
     "yfinance + arch-package GARCH-family fit (BIC-selected)"),
    ("figure_A4_drawdowns", "empirical", "Appendix A", "yfinance (BTC-USD), native 365-day calendar"),
    ("figure_A5_correlation_dcc", "empirical", "Appendix A",
     "yfinance + arch-package DCC-GARCH(1,1) fit, raw and DXY/VIX-residualized"),
    ("figure_B1_wash_trading_updated", "reproduced", "Appendix B",
     "Bitwise (2019); Cong et al. (2022); Sila et al. (2025) (cited)"),
    ("figure_B2_tether_dominance", "reproduced", "Appendix B", "CoinMarketCap (2025); CoinLaw (2026) (cited)"),
    ("figure_C1_attack_cost_breakdown", "reproduced", "Appendix C", "Harvey (2025, 2026) (cited)"),
]


def export_figure_manifest():
    """Machine-readable table classifying every figure by BOTH evidentiary
    status (Type) and where it is actually used (Role) -- so the paper's
    narrowed scope is enforced by the manifest, not just asserted in prose."""
    df = pd.DataFrame(FIGURE_MANIFEST, columns=["Figure", "Type", "Role", "Data_Source"])
    print("\n--- Figure Manifest (evidentiary status and manuscript role) ---")
    print(df.to_string(index=False))
    export_table(df, "table_figure_manifest",
                 caption="Classification of every figure by evidentiary status, manuscript role, "
                         "and data source.",
                 label="tab:figure_manifest")
    return df


# ==============================================================================
# 8. MAIN MENU AND EXECUTION
# ==============================================================================

def main_menu(series_dict, data_loaded):
    menu = {
        '1':  ('Fig 1: The Re-Intermediation Trilemma (framework, NEW)', fig01_reintermediation_trilemma, None),
        '2':  ('Fig 2: Settlement-layer throughput (Pillar I)', fig02_settlement_layer_comparison, None),
        '3':  ('Fig 3: LN topology mesh vs. hub-and-spoke (Pillar II, theory)', fig03_ln_topology_schematic, None),
        '4':  ('Fig 4: LN reliability synthesis (Pillar II, evidence)', fig04_ln_reliability_synthesis, None),
        '5':  ('Fig 5: Security budget dilemma (Pillar III, theory)', fig05_security_budget_dilemma, None),
        '6':  ('Fig 6: Mining pool concentration (Pillar III, evidence)', fig06_mining_pool_concentration, None),
        '7':  ('Fig A1: Volatility comparison (Appendix A)', figA1_rolling_volatility, 'series'),
        '8':  ('Figs A2-A3: VaR/ES suite + GARCH race (Appendix A)', figA2_A3_risk_suite, 'series'),
        '9':  ('Figs A4-A5: Drawdowns + DCC-GARCH correlation (Appendix A)', figA4_A5_digital_gold_robustness, 'series'),
        '10': ('Table A: ADF stationarity report (Appendix A)', tableA_adf_stationarity_report, 'series'),
        '11': ('Fig B1: Wash-trading estimates (Appendix B)', figB1_wash_trading, None),
        '12': ('Fig B2: Tether/stablecoin dominance (Appendix B)', figB2_tether_dominance, None),
        '13': ('Fig C1: 51% attack cost breakdown (Appendix C)', figC1_attack_cost_breakdown, None),
        '14': ('Figure manifest (evidentiary status + manuscript role)', export_figure_manifest, None),
        '15': ('Run all', 'run_all', None),
        '0':  ('Exit', 'exit', None),
    }

    def execute(choice):
        desc, func, needs = menu[choice]
        if needs == 'series' and not data_loaded:
            print(f"\n[ERROR] Cannot run '{desc}': market data unavailable.")
            return
        func(series_dict) if needs == 'series' else func()

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
        print("   v7 ANALYSIS -- RE-INTERMEDIATION TRILEMMA -- MAIN MENU")
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
    print(f"Starting v7 analysis (Re-Intermediation Trilemma), paper date: {FINAL_ANALYSIS_DATE}")
    print(f"Dependency status: yfinance={HAS_YFINANCE}, arch={HAS_ARCH}, statsmodels={HAS_STATSMODELS}")
    missing = [p for p, ok in [("yfinance", HAS_YFINANCE), ("arch", HAS_ARCH),
                                ("statsmodels", HAS_STATSMODELS)] if not ok]
    if missing:
        print(f"  -> Missing: {', '.join(missing)}. Install with: pip install {' '.join(missing)}")

    # Fixed end date (one day past the stated analysis date) rather than
    # datetime.now(), so the script is actually reproducible.
    effective_end_date = (pd.Timestamp(FINAL_ANALYSIS_DATE) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        series_dict = load_all_series(FULL_START_DATE, effective_end_date, CACHE_FILENAME)
        data_loaded = bool(series_dict) and all(len(s) > 0 for s in series_dict.values())
    except (ConnectionError, ValueError) as e:
        print(f"\n[CRITICAL] Could not load market data: {e}")
        series_dict = {}
        data_loaded = False

    main_menu(series_dict, data_loaded)
