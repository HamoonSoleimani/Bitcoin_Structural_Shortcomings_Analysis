# ==============================================================================
#      PYTHON SCRIPT FOR REPRODUCIBLE DATA ANALYSIS AND VISUALIZATION
# ------------------------------------------------------------------------------
# Title:      An Examination of Bitcoin's Structural Shortcomings as Money:
#             A Synthesis of Economic and Technical Critiques
# Author:     Hamoon Soleimani
# Date:       November 13, 2025 (Date of final analysis)
#
# Description: This script reproduces the core quantitative analyses
#              for Figures 2, 3, 4, 5, 11, and 12, ensuring full
#              reproducibility via a static data file.
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model
import datetime
import os
import warnings
import mplcyberpunk

# --- 1. GLOBAL CONFIGURATION & STATIC PARAMETERS ---

# Define global style for most plots
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore", category=UserWarning) # Suppress minor plot warnings

# --- HARDCODED DATES (CRITICAL FOR REPRODUCIBILITY) ---
# All analyses are anchored to this date range.
FINAL_ANALYSIS_DATE = '2025-11-13'
START_DATE_DRAWDOWN = '2015-01-01'
START_DATE_VOLATILITY = '2020-01-01'
FULL_START_DATE = START_DATE_DRAWDOWN

# Define the tickers for the primary assets
TICKERS = {
    'Bitcoin': 'BTC-USD',
    'US Dollar': 'UUP',
    'Gold': 'GC=F',
    'S&P 500': '^GSPC'
}
ASSETS_FOR_VOL_COMP = {"AAPL": "Apple", "BTC-USD": "Bitcoin", "GC=F": "Gold"}
CACHE_FILENAME = "research_data_static.csv"


# --- 2. ROBUST DATA LOADING UTILITY ---

def get_data(tickers_dict, start_date, end_date, cache_filename):
    """
    Loads historical price data. Prioritizes static CSV for reproducibility.
    If CSV is missing, it fetches data from yfinance and creates the CSV.
    """
    tickers = list(tickers_dict.values())
    
    if os.path.exists(cache_filename):
        print(f"Loading data from static file: {cache_filename}...")
        try:
            data = pd.read_csv(cache_filename, index_col='Date', parse_dates=True)
            # Ensure the required columns exist (yfinance sometimes changes names/formats)
            if all(t in data.columns for t in tickers):
                # Ensure date range is correct
                return data.loc[start_date:end_date]
            else:
                print("Static data file is corrupt or missing required columns. Refetching.")
        except Exception as e:
            print(f"Error loading static file ({e}). Refetching data.")

    # Fallback to yfinance (downloads the data used for the analysis)
    print("Fetching new data from yfinance (API Fallback)...")
    try:
        data = yf.download(tickers, start=FULL_START_DATE, end=end_date)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        
        # Add Apple data specifically for Figure 2
        aapl_data = yf.download(ASSETS_FOR_VOL_COMP['AAPL'], start=FULL_START_DATE, end=end_date)['Close']
        data['AAPL'] = aapl_data
        
        data.to_csv(cache_filename) # Save for future runs
        print(f"Data saved to static file: {cache_filename}")
        return data.loc[start_date:end_date].dropna()
    except Exception as e:
        raise ConnectionError(f"Failed to fetch data from yfinance: {e}. Check network connection or API status.")


# --- 3. ANALYSIS AND VISUALIZATION FUNCTIONS ---

def generate_volatility_comparison_chart(df_raw):
    """
    Reproduces Figure 2: Comparative Rolling Volatility (BTC, Gold, Apple).
    """
    plt.style.use("cyberpunk")
    ROLLING_WINDOWS = [15, 200]
    df = df_raw.rename(columns=ASSETS_FOR_VOL_COMP).dropna()
    
    # Calculate returns and rolling volatility
    for col in ASSETS_FOR_VOL_COMP.values():
        df[f'{col} Returns'] = df[col].pct_change()
        for window in ROLLING_WINDOWS:
            df[f'{col} Volatility {window}d'] = df[f'{col} Returns'].rolling(window).std() * np.sqrt(252) # Annualized

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    colors = {'Apple': 'c', 'Bitcoin': 'm', 'Gold': 'y'}
    
    ax1 = axes[0]
    for asset_name in ASSETS_FOR_VOL_COMP.values():
        df[f'{asset_name} Volatility 15d'].plot(ax=ax1, color=colors[asset_name], lw=2, label=asset_name)
    ax1.set_ylabel("15-Day Annualized Volatility")
    ax1.set_title("Short-Term Volatility Comparison (Annualized)")
    ax1.legend()
    
    ax2 = axes[1]
    for asset_name in ASSETS_FOR_VOL_COMP.values():
        df[f'{asset_name} Volatility 200d'].plot(ax=ax2, color=colors[asset_name], lw=2, label=asset_name)
    ax2.set_ylabel("200-Day Annualized Volatility")
    ax2.set_title("Long-Term Volatility Comparison (Annualized)")
    
    fig.suptitle("Comparative Rolling Volatility: Bitcoin, Gold, and Apple (Figure 2)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('figure_2_rolling_volatility.png', dpi=300)
    plt.show()
    plt.style.use('seaborn-v0_8-whitegrid') # Revert style


def analyze_risk_and_garch(data):
    """
    Runs VaR analysis (Figure 3) and GARCH modeling (Figure 4) on asset returns.
    """
    log_returns = np.log(data / data.shift(1)).dropna()
    
    # --- Value-at-Risk (VaR) Analysis for Figure 3 ---
    # Calculate 1-day 95% VaR (5th percentile of historical log returns)
    var_results = [{'Asset': name, 'VaR_95': log_returns[ticker].quantile(0.05) * 100}
                   for name, ticker in TICKERS.items() if ticker in log_returns.columns]
    var_df = pd.DataFrame(var_results).sort_values('VaR_95', ascending=True)
    
    fig_var, ax_var = plt.subplots(figsize=(10, 6))
    bars = ax_var.bar(var_df['Asset'], var_df['VaR_95'], color=['#e74c3c', '#c0392b', '#95a5a6', '#7f8c8d'])
    ax_var.set_title('1-Day 95% Value-at-Risk (VaR) Comparison (Figure 3)', fontsize=16)
    ax_var.set_ylabel('Potential 1-Day Loss (%)')
    
    for bar in bars:
        yval = bar.get_height()
        ax_var.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f'{yval:.2f}%', ha='center', va='bottom', weight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_3_value_at_risk.png', dpi=300)
    plt.show()

    # --- GARCH Modeling for Figure 4 ---
    btc_returns = log_returns[TICKERS['Bitcoin']].dropna() * 100
    # Use standard GARCH(1,1) model with t-distribution error assumption 
    # for fat tails, typical for crypto data (as suggested by literature).
    model = arch_model(btc_returns, vol='Garch', p=1, q=1, dist='t')
    results = model.fit(disp='off')
    
    print("\nGARCH(1,1) Model Parameters (Figure 4 supporting data):")
    print(results.summary())
    
    # Calculate volatility persistence (alpha + beta)
    persistence = results.params['alpha[1]'] + results.params['beta[1]']
    half_life = np.log(0.5) / np.log(persistence)
    print(f"\nVolatility Persistence (alpha + beta): {persistence:.4f}")
    print(f"Volatility Half-Life (days): {half_life:.1f}")

    fig_garch, ax_garch = plt.subplots(figsize=(12, 6))
    ax_garch.plot(btc_returns.index, btc_returns, color='grey', alpha=0.6, label='Daily Returns (%)')
    ax_garch.plot(results.conditional_volatility.index, results.conditional_volatility, color='red', label='GARCH Conditional Volatility')
    ax_garch.set_title('Bitcoin Daily Returns and GARCH(1,1) Conditional Volatility (Figure 4)', fontsize=16)
    ax_garch.set_ylabel('Percentage (%)')
    ax_garch.legend()
    plt.tight_layout()
    plt.savefig('figure_4_garch_volatility.png', dpi=300)
    plt.show()


def generate_tps_chart():
    """
    Reproduces Figure 5: Transaction Per Second (TPS) Capacity Comparison.
    Data is hardcoded from corporate 10-K filings and public ledger data.
    """
    btc_tps = 6.0  # Realized average Bitcoin on-chain TPS
    mastercard_transactions_2024 = 159.4e9
    visa_transactions_2024 = 303e9
    seconds_in_year = 365.25 * 24 * 60 * 60
    
    mastercard_tps = mastercard_transactions_2024 / seconds_in_year
    visa_tps = visa_transactions_2024 / seconds_in_year
    
    systems = ['Bitcoin\n(Realized Avg)', 'Mastercard\n(Switched)', 'Visa\n(Total Payments)']
    tps_values = [btc_tps, mastercard_tps, visa_tps]
    colors = ['#f2a900', '#eb001b', '#1a1f71']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(systems, tps_values, color=colors)
    ax.set_yscale('log')
    ax.set_title('Transaction Per Second (TPS) Capacity Comparison (Log Scale) (Figure 5)', fontsize=16, pad=20)
    ax.set_ylabel('Transactions Per Second (Log Scale)', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        # Annotate with the precise value
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    plt.savefig('figure_5_tps_comparison.png', dpi=300)
    plt.show()


def analyze_digital_gold_narrative(data):
    """
    Generates plots related to the 'digital gold' narrative:
    Drawdown Analysis (Figure 11) and Correlation Analysis (Figure 12).
    """
    
    # --- Drawdown Analysis for Figure 11 ---
    btc_price = data[TICKERS['Bitcoin']].dropna()
    previous_peaks = btc_price.cummax()
    drawdowns = (btc_price - previous_peaks) / previous_peaks
    
    fig_draw, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(btc_price.index, btc_price, label='Bitcoin Price (USD)', color='blue')
    ax1.set_yscale('log')
    ax1.set_title('Bitcoin Price and Historical Drawdowns (Figure 11)', fontsize=16)
    ax1.set_ylabel('Price (USD, Log Scale)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--")
    
    ax2.plot(drawdowns.index, drawdowns * 100, label='Drawdown', color='red')
    ax2.fill_between(drawdowns.index, drawdowns * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    
    # Highlight max drawdown percentage
    max_dd = drawdowns.min() * 100
    ax2.text(drawdowns.idxmin(), max_dd, f'Max DD: {max_dd:.1f}%', ha='right', va='top', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure_11_drawdowns.png', dpi=300)
    plt.show()
    
    # --- Correlation Analysis for Figure 12 ---
    sp500_ticker = TICKERS['S&P 500']
    btc_ticker = TICKERS['Bitcoin']
    
    log_returns = np.log(data[[btc_ticker, sp500_ticker]] / data[[btc_ticker, sp500_ticker]].shift(1)).dropna()
    rolling_corr = log_returns[btc_ticker].rolling(window=60).corr(log_returns[sp500_ticker])
    
    fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
    rolling_corr.plot(ax=ax_corr, color='purple')
    ax_corr.set_title('60-Day Rolling Correlation: Bitcoin vs. S&P 500 (Figure 12)', fontsize=16)
    ax_corr.set_ylabel('Pearson Correlation Coefficient')
    ax_corr.axhline(0, color='black', linestyle='--', lw=1)
    
    # Highlight periods of high correlation (e.g., 2022)
    peak_corr = rolling_corr.max()
    ax_corr.text(rolling_corr.idxmax(), peak_corr, f'Peak: {peak_corr:.2f}', ha='center', va='bottom', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figure_12_rolling_correlation.png', dpi=300)
    plt.show()


# --- 4. SCRIPT EXECUTION ---

if __name__ == '__main__':
    print(f"Starting analysis using data up to: {FINAL_ANALYSIS_DATE}")
    
    # 1. Load Data
    full_data = get_data(TICKERS, start_date=FULL_START_DATE, end_date=FINAL_ANALYSIS_DATE, cache_filename=CACHE_FILENAME)
    
    # Define subsets based on analysis start dates
    volatility_data = full_data.loc[START_DATE_VOLATILITY:]
    
    if full_data.empty or volatility_data.empty:
        print("ERROR: Data frames are empty. Cannot proceed with analysis.")
    else:
        # 2. Generate Figures
        print("\n--- Generating Figures ---")
        generate_volatility_comparison_chart(full_data)
        analyze_risk_and_garch(volatility_data)
        generate_tps_chart()
        analyze_digital_gold_narrative(full_data)
        
        print("\nAll data-driven figures have been generated and saved to the current directory.")
        print("Note: GARCH parameters and Volatility Persistence/Half-Life are printed above.")
