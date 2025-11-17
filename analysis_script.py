# ==============================================================================
#      PYTHON SCRIPT FOR REPRODUCIBLE DATA ANALYSIS AND VISUALIZATION
# ------------------------------------------------------------------------------
# Title:      An Examination of Bitcoin's Structural Shortcomings as Money:
#             A Synthesis of Economic and Technical Critiques
# Author:     Hamoon Soleimani
# Date:       November 13, 2025 (Date of final analysis)
#
# Description: This script reproduces the core quantitative analyses and
#              visualizations for the research paper, including Figures 2, 3, 4,
#              5, 6, 9, 17, 18, 20, and 22. It ensures full reproducibility for
#              data-driven plots via a static data file.
# Version:    7.0 (Final - Added Figure 22 and refactored menu)
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from arch import arch_model
import datetime
import os
import warnings

# --- 1. GLOBAL CONFIGURATION & STATIC PARAMETERS ---

# --- PLOT STYLE CONFIGURATION ---
# A custom, readable dark theme for all plots.
plt.style.use('dark_background')
plt.rcParams.update({
    "grid.color": "#555555",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "text.color": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "legend.facecolor": "#1c1c1c",
    "legend.edgecolor": "white",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})
warnings.filterwarnings("ignore", category=UserWarning) # Suppress minor plot warnings

# --- HARDCODED DATES (CRITICAL FOR REPRODUCIBILITY) ---
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

def get_data(start_date, end_date, cache_filename):
    """
    Loads historical price data. Prioritizes static CSV for reproducibility.
    If CSV is missing, it fetches all required data in a single API call.
    """
    all_tickers_needed = list(TICKERS.values()) + list(ASSETS_FOR_VOL_COMP.keys())
    all_tickers_needed = sorted(list(set(all_tickers_needed)))

    if os.path.exists(cache_filename):
        print(f"Loading data from static file: {cache_filename}...")
        try:
            data = pd.read_csv(cache_filename, index_col='Date', parse_dates=True)
            if not data.empty and all(t in data.columns for t in all_tickers_needed):
                return data.dropna()
            else:
                print("Static data file is empty or incomplete. Refetching all data.")
        except Exception as e:
            print(f"Error loading static file ({e}). Refetching data.")

    print("Fetching new data from yfinance (API Fallback)...")
    try:
        data = yf.download(all_tickers_needed, start=start_date, end=end_date)['Close'].dropna()
        if data.empty:
            raise ValueError("Download from yfinance resulted in an empty DataFrame. Check tickers and dates.")
        
        data.to_csv(cache_filename)
        print(f"Data saved to static file: {cache_filename}")
        return data
    except Exception as e:
        raise ConnectionError(f"Failed to fetch data from yfinance: {e}")


# --- 3. ANALYSIS AND VISUALIZATION FUNCTIONS ---

def generate_volatility_comparison_chart(df_raw):
    """
    Reproduces Figure 2: Comparative Rolling Volatility (BTC, Gold, Apple).
    """
    print("\nGenerating Figure 2: Comparative Rolling Volatility...")
    ROLLING_WINDOWS = [15, 200]
    df = df_raw.rename(columns=ASSETS_FOR_VOL_COMP).dropna()
    
    for col in ASSETS_FOR_VOL_COMP.values():
        df[f'{col} Returns'] = df[col].pct_change()
        for window in ROLLING_WINDOWS:
            df[f'{col} Volatility {window}d'] = df[f'{col} Returns'].rolling(window).std() * np.sqrt(252)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)
    colors = {'Apple': '#3498db', 'Bitcoin': '#f1c40f', 'Gold': '#e74c3c'}
    
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


def analyze_risk_and_garch(data):
    """
    Runs VaR analysis (Figure 3) and GARCH modeling (Figure 4) on asset returns.
    """
    print("\nGenerating Figure 3: Value-at-Risk (VaR) Comparison...")
    log_returns = np.log(data / data.shift(1)).dropna()
    
    var_results = [{'Asset': name, 'VaR_95': log_returns[ticker].quantile(0.05) * 100}
                   for name, ticker in TICKERS.items() if ticker in log_returns.columns]
    var_df = pd.DataFrame(var_results).sort_values('VaR_95', ascending=True)
    
    fig_var, ax_var = plt.subplots(figsize=(10, 6))
    bar_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    bars = ax_var.bar(var_df['Asset'], var_df['VaR_95'], color=bar_colors)
    ax_var.set_title('1-Day 95% Value-at-Risk (VaR) Comparison (Figure 3)', fontsize=16)
    ax_var.set_ylabel('Potential 1-Day Loss (%)')
    
    for bar in bars:
        yval = bar.get_height()
        ax_var.text(bar.get_x() + bar.get_width() / 2.0, yval - 0.1, f'{yval:.2f}%', ha='center', va='top', weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('figure_3_value_at_risk.png', dpi=300)
    plt.show()

    print("\nGenerating Figure 4: GARCH Conditional Volatility...")
    btc_returns = log_returns[TICKERS['Bitcoin']].dropna() * 100
    model = arch_model(btc_returns, vol='Garch', p=1, q=1, dist='t')
    results = model.fit(disp='off')
    
    print("\nGARCH(1,1) Model Parameters (Figure 4 supporting data):")
    print(results.summary())
    
    persistence = results.params['alpha[1]'] + results.params['beta[1]']
    half_life = np.log(0.5) / np.log(persistence)
    print(f"\nVolatility Persistence (alpha + beta): {persistence:.4f}")
    print(f"Volatility Half-Life (days): {half_life:.1f}")

    fig_garch, ax_garch = plt.subplots(figsize=(12, 6))
    ax_garch.plot(btc_returns.index, btc_returns, color='grey', alpha=0.6, label='Daily Returns (%)')
    ax_garch.plot(results.conditional_volatility.index, results.conditional_volatility, color='#e74c3c', label='GARCH Conditional Volatility')
    ax_garch.set_title('Bitcoin Daily Returns and GARCH(1,1) Conditional Volatility (Figure 4)', fontsize=16)
    ax_garch.set_ylabel('Percentage (%)')
    ax_garch.legend()
    plt.tight_layout()
    plt.savefig('figure_4_garch_volatility.png', dpi=300)
    plt.show()


def generate_supply_volatility_model_figure_dark():
    """
    Reproduces Figure 5: A theoretical model comparing fixed vs. elastic supply.
    This version is redesigned to be fully compatible with the script's dark theme.
    """
    print("\nGenerating Figure 5: Supply/Demand Model...")
    # Create figure using global dark theme settings
    fig = plt.figure(figsize=(18, 9), dpi=150)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.95, top=0.90, bottom=0.12, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def setup_axes(ax, title):
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        arrow_props = dict(arrowstyle='->', lw=2, color='white', mutation_scale=20, clip_on=False)
        ax.add_patch(FancyArrowPatch((0, 0), (11.5, 0), **arrow_props))
        ax.add_patch(FancyArrowPatch((0, 0), (0, 11.5), **arrow_props))
        ax.text(11.8, -0.4, 'Quantity ($q$)', fontsize=13, ha='center', va='top', style='italic', weight='semibold')
        ax.text(-0.5, 11.8, 'Price ($p$)', fontsize=13, ha='right', va='center', style='italic', weight='semibold')
        ax.text(5.5, 12.8, title, fontsize=17, ha='center', va='bottom', weight='bold', 
                bbox=dict(boxstyle='round,pad=0.6', facecolor='#1a1a1a', edgecolor='white', alpha=0.95, linewidth=1.5))

    def draw_bracket(ax, x1, y1, x2, y2, label, o='vertical', lo=0.5, c='white', cap_inward=False):
        props = dict(color=c, lw=1.5, solid_capstyle='round'); offset = 0.2 if cap_inward else -0.2
        if o == 'vertical':
            ax.plot([x1, x1], [y1, y2], **props); ax.plot([x1, x1 + offset], [y1, y1], **props); ax.plot([x1, x1 + offset], [y2, y2], **props)
            ax.text(x1 - lo, (y1 + y2) / 2, label, ha='right', va='center', fontsize=11, style='italic', weight='semibold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', edgecolor=c, alpha=0.95, linewidth=1.2))
        else:
            ax.plot([x1, x2], [y1, y1], **props); ax.plot([x1, x1], [y1, y1 + offset], **props); ax.plot([x2, x2], [y1, y1 + offset], **props)
            ax.text((x1 + x2) / 2, y1 - lo, label, ha='center', va='top', fontsize=11, style='italic', weight='semibold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='black', edgecolor=c, alpha=0.95, linewidth=1.2))

    # Chart 1: Bitcoin
    setup_axes(ax1, 'Bitcoin (Fixed Supply)')
    q_s = 5.5; ax1.plot([q_s, q_s], [0, 11], '#3498db', lw=3, zorder=3)
    ax1.text(q_s + 0.4, 11.5, '$S$', ha='center', va='bottom', fontsize=16, weight='bold', color='#3498db', bbox=dict(boxstyle='round,pad=0.35', facecolor='black', edgecolor='#3498db', lw=1.5))
    dx=np.linspace(0, 8.5, 200); dy1=11.5-dx; dy2=8.5-dx
    ax1.plot(dx, dy1,'-',c='#e74c3c',lw=3,zorder=2); ax1.plot(dx,dy2,'--',c='#f39c12',lw=3,dashes=(8,4),zorder=2)
    ax1.text(8.7,3.0,'$D$',fontsize=16,weight='bold',c='#e74c3c',bbox=dict(boxstyle='circle,pad=0.4',facecolor='black',edgecolor='#e74c3c',lw=1.5))
    ax1.text(8.7,0.2,"$D'$",fontsize=16,weight='bold',c='#f39c12',bbox=dict(boxstyle='circle,pad=0.4',facecolor='black',edgecolor='#f39c12',lw=1.5))
    p1,p2=6.0,3.0; ax1.plot([0,q_s],[p1,p1],c='#777',ls='--',lw=1.5,dashes=(6,4),zorder=1); ax1.plot([0,q_s],[p2,p2],c='#777',ls='--',lw=1.5,dashes=(6,4),zorder=1)
    ax1.plot(q_s,p1,'o',ms=11,c='#e74c3c',mec='white',mew=2.5,zorder=5); ax1.plot(q_s,p2,'o',ms=11,c='#f39c12',mec='white',mew=2.5,zorder=5)
    ax1.text(q_s+0.6, p1, '$E_1$', fontsize=12, weight='bold'); ax1.text(q_s+0.6, p2, '$E_2$', fontsize=12, weight='bold')
    bbox_props = dict(boxstyle='round,pad=0.25',facecolor='black',edgecolor='#777',lw=0.8)
    ax1.text(-0.4,p1,'$p_1$',ha='right',va='center',fontsize=12,style='italic',weight='semibold',bbox=bbox_props)
    ax1.text(-0.4,p2,'$p_2$',ha='right',va='center',fontsize=12,style='italic',weight='semibold',bbox=bbox_props)
    ax1.text(q_s,-0.4,'$q^*$',ha='center',va='top',fontsize=12,style='italic',weight='semibold',bbox=bbox_props)
    draw_bracket(ax1,-1.1,p2,-1.1,p1,'Price\nVolatility',c='#e74c3c',cap_inward=True); bx=7.5; draw_bracket(ax1,bx+0.3,8.5-bx,bx+0.3,11.5-bx,'Demand\nShift','vertical',-0.8,'#bbb')

    # Chart 2: Alternative
    setup_axes(ax2, 'Alternative System (Elastic Supply)')
    p_s = 5.5; ax2.plot([0,11],[p_s,p_s],'#3498db',lw=3,zorder=3)
    ax2.text(11.5, p_s+0.3,'$S$',ha='left',va='bottom',fontsize=16,weight='bold',color='#3498db',bbox=dict(boxstyle='round,pad=0.35',facecolor='black',edgecolor='#3498db',lw=1.5))
    ax2.plot(dx,dy1,'-',c='#e74c3c',lw=3,zorder=2); ax2.plot(dx,dy2,'--',c='#f39c12',lw=3,dashes=(8,4),zorder=2)
    ax2.text(8.7,3.0,'$D$',fontsize=16,weight='bold',c='#e74c3c',bbox=dict(boxstyle='circle,pad=0.4',facecolor='black',edgecolor='#e74c3c',lw=1.5))
    ax2.text(8.7,0.2,"$D'$",fontsize=16,weight='bold',c='#f39c12',bbox=dict(boxstyle='circle,pad=0.4',facecolor='black',edgecolor='#f39c12',lw=1.5))
    q1,q2=6.0,3.0; ax2.plot([q1,q1],[0,p_s],c='#777',ls='--',lw=1.5,dashes=(6,4),zorder=1); ax2.plot([q2,q2],[0,p_s],c='#777',ls='--',lw=1.5,dashes=(6,4),zorder=1)
    ax2.plot(q1,p_s,'o',ms=11,c='#e74c3c',mec='white',mew=2.5,zorder=5); ax2.plot(q2,p_s,'o',ms=11,c='#f39c12',mec='white',mew=2.5,zorder=5)
    ax2.text(q1,p_s+0.6,'$E_1$',fontsize=12,weight='bold',ha='center',va='bottom'); ax2.text(q2,p_s+0.6,'$E_2$',fontsize=12,weight='bold',ha='center',va='bottom')
    ax2.text(q1,-0.4,'$q_1$',ha='center',va='top',fontsize=12,style='italic',weight='semibold',bbox=props)
    ax2.text(q2,-0.4,'$q_2$',ha='center',va='top',fontsize=12,style='italic',weight='semibold',bbox=props)
    ax2.text(-0.4,p_s,'$p^*$',ha='right',va='center',fontsize=12,style='italic',weight='semibold',bbox=props)
    ax2.plot([-1.3,-0.2],[p_s,p_s],'-',c='#2ecc71',lw=3,zorder=4); ax2.text(-1.5,p_s,'Price\nStability',ha='right',va='center',fontsize=11,style='italic',c='#2ecc71',weight='bold',bbox=dict(boxstyle='round,pad=0.4',facecolor='#102510',edgecolor='#2ecc71',alpha=0.95,lw=1.5))
    draw_bracket(ax2,q2,-1.3,q1,-1.3,'Supply\nAdjustment','horizontal',0.85,'#3498db',cap_inward=True); draw_bracket(ax2,bx+0.3,8.5-bx,bx+0.3,11.5-bx,'Demand\nShift','vertical',-0.8,'#bbb')
    
    # Caption and final touches
    cy=0.04;
    fig.text(0.5,cy-0.01,'Comparison of Fixed vs. Flexible Supply Response',ha='center',va='top',fontsize=13,weight='semibold')
    fig.text(0.5,cy-0.04,'Left: Inelastic supply, price volatility | Right: Elastic supply, price stability',ha='center',va='top',fontsize=11,style='italic',c='#aaa')
    for ax in [ax1, ax2]: ax.grid(True,alpha=0.12,ls=':',lw=0.6,zorder=0); ax.set_axisbelow(True)
    plt.savefig('figure_5_supply_volatility_model.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()


def generate_tps_chart():
    """
    Reproduces Figure 6: Transaction Per Second (TPS) Capacity Comparison.
    """
    print("\nGenerating Figure 6: TPS Capacity Comparison...")
    btc_tps = 6.0
    mastercard_transactions_2024 = 159.4e9
    visa_transactions_2024 = 303e9
    seconds_in_year = 365.25 * 24 * 60 * 60
    
    mastercard_tps = mastercard_transactions_2024 / seconds_in_year
    visa_tps = visa_transactions_2024 / seconds_in_year
    
    systems = ['Bitcoin\n(Realized Avg)', 'Mastercard\n(Switched)', 'Visa\n(Total Payments)']
    tps_values = [btc_tps, mastercard_tps, visa_tps]
    colors = ['#f2a900', '#eb001b', '#3498db']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(systems, tps_values, color=colors)
    ax.set_yscale('log')
    ax.set_title('Transaction Per Second (TPS) Capacity Comparison (Log Scale) (Figure 6)', fontsize=16, pad=20)
    ax.set_ylabel('Transactions Per Second (Log Scale)', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:,.0f}', ha='center', va='bottom', fontsize=10, weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('figure_6_tps_comparison.png', dpi=300)
    plt.show()


def generate_centralization_parameter_map():
    """
    Reproduces Figure 9: Parameter map for social optimum in the Lightning Network game.
    """
    print("\nGenerating Figure 9: Lightning Network Centralization Map...")
    b_vals = np.linspace(0, 2, 500); c_vals = np.linspace(0, 2, 500)
    B, C = np.meshgrid(b_vals, c_vals)
    Z = np.zeros_like(B)
    Z[C < B] = 0; Z[(C >= B) & (C <= B + 0.5)] = 1; Z[C > B + 0.5] = 2
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#ffeda0', '#a1d99b', '#9ecae1']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    ax.imshow(Z, origin='lower', extent=[0, 2, 0, 2], cmap=cmap, aspect='auto', interpolation='nearest')
    ax.plot(b_vals, b_vals, color='white', linestyle='--', linewidth=1.5, label=r'Boundary: $c = b$')
    ax.plot(b_vals, b_vals + 0.5, color='white', linestyle='-.', linewidth=1.5, label=r'Boundary: $c = b + 0.5$')
    ax.set_xlabel('Incentive to Earn Routing Fees (b)'); ax.set_ylabel('Incentive for Low-Cost Personal Transactions (c)')
    ax.set_title("Social Optimum Topologies in a Payment Network Creation Game (Figure 9)", fontsize=16, pad=15)
    legend_patches = [mpatches.Patch(color=colors[2], label=r'Complete Graph ($c > b + 0.5$)'),
                      mpatches.Patch(color=colors[1], label=r'Star Graph ($b \leq c \leq b + 0.5$)'),
                      mpatches.Patch(color=colors[0], label=r'Path Graph ($c < b$)')]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_patches, loc='upper left')
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.grid(True, which="both", ls=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figure_9_centralization_parameter_map.png", dpi=300)
    plt.show()


def analyze_digital_gold_narrative(data):
    """
    Generates plots related to the 'digital gold' narrative:
    Drawdown Analysis (Figure 17) and Correlation Analysis (Figure 18).
    """
    print("\nGenerating Figure 17: Bitcoin Price and Historical Drawdowns...")
    btc_price = data[TICKERS['Bitcoin']].dropna()
    previous_peaks = btc_price.cummax()
    drawdowns = (btc_price - previous_peaks) / previous_peaks
    
    fig_draw, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.plot(btc_price.index, btc_price, label='Bitcoin Price (USD)', color='#3498db')
    ax1.set_yscale('log'); ax1.set_title('Bitcoin Price and Historical Drawdowns (Figure 17)', fontsize=16)
    ax1.set_ylabel('Price (USD, Log Scale)'); ax1.legend(); ax1.grid(True, which="both", ls="--", color="#555555")
    
    ax2.plot(drawdowns.index, drawdowns * 100, label='Drawdown', color='#e74c3c')
    ax2.fill_between(drawdowns.index, drawdowns * 100, 0, color='#e74c3c', alpha=0.4)
    ax2.set_ylabel('Drawdown (%)'); ax2.set_xlabel('Date')
    max_dd = drawdowns.min() * 100
    ax2.text(drawdowns.idxmin(), max_dd, f'Max DD: {max_dd:.1f}%', ha='right', va='top', color='white', fontsize=10)
    
    plt.tight_layout(); plt.savefig('figure_17_drawdowns.png', dpi=300); plt.show()
    
    print("\nGenerating Figure 18: Bitcoin vs. S&P 500 Rolling Correlation...")
    sp500_ticker = TICKERS['S&P 500']; btc_ticker = TICKERS['Bitcoin']
    log_returns = np.log(data[[btc_ticker, sp500_ticker]] / data[[btc_ticker, sp500_ticker]].shift(1)).dropna()
    rolling_corr = log_returns[btc_ticker].rolling(window=60).corr(log_returns[sp500_ticker])
    
    fig_corr, ax_corr = plt.subplots(figsize=(12, 6))
    rolling_corr.plot(ax=ax_corr, color='#9b59b6')
    ax_corr.set_title('60-Day Rolling Correlation: Bitcoin vs. S&P 500 (Figure 18)', fontsize=16)
    ax_corr.set_ylabel('Pearson Correlation Coefficient'); ax_corr.axhline(0, color='white', linestyle='--', lw=1)
    peak_corr = rolling_corr.max()
    ax_corr.text(rolling_corr.idxmax(), peak_corr, f'Peak: {peak_corr:.2f}', ha='center', va='bottom', color='white', fontsize=10)
    
    plt.tight_layout(); plt.savefig('figure_18_rolling_correlation.png', dpi=300); plt.show()

def generate_oceanic_games_model():
    """
    Reproduces Figure 20: Economic incentive for mining centralization from an
    Oceanic Games model, adapted to a non-linear representation.
    """
    print("\nGenerating Figure 20: Economic Incentive for Mining Centralization...")

    # --- 1. Define the Corrected, Non-Linear Conceptual Model ---
    # The user correctly noted the relationship is non-linear. We will now use
    # quadratic functions to model the curvature seen in the original paper's figure.
    # This better represents the accelerating incentive to centralize.

    def get_coalition_value_per_unit_nonlinear(r):
        """
        Models the increasing and convex (upward-curving) value for the coalition.
        This shows that the strategic advantage accelerates as the coalition grows.
        f(r) = a*r^2 + b*r + c
        """
        # Parameters are chosen to start at 1 and curve up to approx. 1.6 at r=40.
        a = 0.00015  # Positive 'a' for upward (convex) curve
        b = 0.01     # Initial positive slope
        c = 1.0      # Starting value at r=0
        return a * r**2 + b * r + c

    def get_oceanic_value_per_unit_nonlinear(r):
        """
        Models the decreasing and concave (downward-curving) value for oceanic miners.
        This shows their strategic position eroding at an accelerating rate.
        f(r) = a*r^2 + b*r + c
        """
        # Parameters are chosen to start at 1 and curve down to approx. 0.6 at r=40.
        a = -0.00015 # Negative 'a' for downward (concave) curve
        b = -0.005   # Initial negative slope
        c = 1.0      # Starting value at r=0
        return a * r**2 + b * r + c

    # --- 2. Generate Data for Plotting ---
    # The x-axis represents the percentage of total hash rate controlled by the coalition.
    r_crystallized = np.linspace(0, 40, 400)

    # Calculate y-values using the new non-linear model functions.
    v1_values = get_coalition_value_per_unit_nonlinear(r_crystallized)
    voc_values = get_oceanic_value_per_unit_nonlinear(r_crystallized)

    # --- 3. Create the Visualization ---
    # Note: The global dark theme from the script's configuration is used automatically.
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the coalition's value (v1) in red
    ax.plot(r_crystallized, v1_values,
            color='#d62728',
            linewidth=2.5,
            label=r'$v_1$ (Value for Coalition Members)')

    # Plot the oceanic miners' value (voc) in blue
    ax.plot(r_crystallized, voc_values,
            color='#1f77b4',
            linewidth=2.5,
            label=r'$v_{oc}$ (Value for Oceanic Miners)')

    # --- 4. Style and Format the Plot ---
    ax.set_xlabel('Percentage of Total Hash Rate in New Coalition ($r_1$)', fontsize=12)
    ax.set_ylabel('Strategic Value Per Unit of Hash Rate', fontsize=12)
    ax.set_title('Economic Incentive for Mining Centralization (Figure 20)', fontsize=14, weight='bold')

    # Set axis limits
    ax.set_xlim(0, 40)
    ax.set_ylim(0.5, 1.7)

    # Add a legend
    legend = ax.legend(fontsize=11, title="Value per Unit of Resource")
    legend.get_title().set_fontweight('bold')

    # Improve tick label appearance
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Display the plot
    plt.tight_layout()

    # --- 5. Save the Figure ---
    plt.savefig('figure_20_centralization_incentive.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

def generate_security_budget_dilemma_chart():
    """
    Reproduces Figure 22: A model of the Bitcoin Security Budget Dilemma.
    This chart is conceptual and does not use external market data.
    """
    print("\nGenerating Figure 22: Bitcoin Security Budget Dilemma...")

    # --- 1. Setup the Plot Figure using global dark theme ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors compatible with the dark theme
    color_blue = '#56B4E9' # A brighter blue
    color_red = '#D55E00'  # A brighter orange-red

    # --- 2. Define Conceptual Data Points ---
    # X-axis represents time points for events
    x_events = {
        "Present": 0,
        "2028 Halving": 1.5,
        "2032 Halving": 3.0,
        "...Post-Subsidy Era": 4.5
    }
    x_ticks = list(x_events.values())
    x_labels = list(x_events.keys())

    # Y-axis represents qualitative security budget levels
    y_levels = {
        "Vulnerable": 0,
        "Low": 1,
        "Medium": 2,
        "High": 3
    }
    y_ticks = list(y_levels.values())
    y_labels = list(y_levels.keys())

    # Data for the BLUE dashed line (Block Subsidy / Scenario A outcome)
    x_subsidy = [0, 1.5, 1.5, 3.0, 3.0, 4.5, 4.5, 5.5]
    y_subsidy = [3, 3,   2,   2,   1,   1,   0.4, 0.2]

    # Data for the RED solid line (Total Security in Scenario B)
    x_l1_retention = [0, 1.5, 1.5, 5.5]
    y_l1_retention = [3, 3,   2.8, 2.5]

    # --- 3. Plot the Data Series ---

    # Plot Scenario A (Blue Dashed Line)
    ax.plot(x_subsidy, y_subsidy, 
            linestyle='--', 
            color=color_blue,
            lw=2.5, 
            label='Scenario A: High L2 Adoption')

    # Plot Scenario B (Red Solid Line)
    ax.plot(x_l1_retention, y_l1_retention, 
            linestyle='-', 
            color=color_red,
            lw=2.5,
            label='Scenario B: L1 Retention')

    # --- 4. Customize Axes, Ticks, and Labels ---

    ax.set_xlabel("Time", fontsize=14, labelpad=10)
    ax.set_ylabel("Security Budget", fontsize=14, labelpad=10)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=12)

    ax.set_xlim(-0.2, 5.7)
    ax.set_ylim(-0.2, 3.5)

    # Spines are handled by the global dark theme, but we can ensure top/right are off
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- 5. Add Annotations and Legend ---

    ax.annotate(
        'Requires Prohibitively High\nTransaction Fees',
        xy=(2.25, 2.75),
        xytext=(2.5, 3.1),
        fontsize=11,
        color=color_red,
        ha='left',
        arrowprops=dict(facecolor=color_red, shrink=0.05, width=1, headwidth=6, edgecolor='none')
    )

    ax.annotate(
        'Transaction Fees',
        xy=(4.0, 2.55),
        xytext=(4.0, 1.5),
        fontsize=11,
        ha='center',
        va='center',
        arrowprops=dict(
            arrowstyle='<->',
            lw=1.5,
            color='white', # Changed from black to white for visibility
            shrinkA=5,
            shrinkB=5
        )
    )

    blue_patch = mpatches.Patch(color=color_blue, label='Scenario A: High L2 Adoption')
    red_patch = mpatches.Patch(color=color_red, label='Scenario B: L1 Retention')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=12, frameon=True)

    # --- 6. Add Title ---

    ax.set_title("A Model of the Bitcoin Security Budget Dilemma", fontsize=16, pad=20, weight='bold')

    # --- 7. Final Touches and Display ---

    plt.tight_layout()
    plt.savefig('figure_22_security_budget_dilemma.png', dpi=300)
    plt.show()


# --- 4. MAIN MENU AND SCRIPT EXECUTION ---

def main_menu(full_data, volatility_data, data_loaded):
    """Displays the main menu and handles user choices for figure generation."""

    menu_options = {
        '1': ('Figure 2: Comparative Rolling Volatility', generate_volatility_comparison_chart, 'full'),
        '2': ('Figures 3 & 4: VaR and GARCH Analysis', analyze_risk_and_garch, 'volatility'),
        '3': ('Figure 5: Supply/Demand Model (Fixed vs. Elastic)', generate_supply_volatility_model_figure_dark, None),
        '4': ('Figure 6: TPS Capacity Comparison', generate_tps_chart, None),
        '5': ('Figure 9: LN Centralization Parameter Map', generate_centralization_parameter_map, None),
        '6': ('Figures 17 & 18: Drawdown and Correlation Analysis', analyze_digital_gold_narrative, 'full'),
        '7': ('Figure 20: Economic Incentive for Mining Centralization', generate_oceanic_games_model, None),
        '8': ('Figure 22: Bitcoin Security Budget Dilemma', generate_security_budget_dilemma_chart, None),
        '9': ('Run All Figures', 'run_all', None),
        '0': ('Exit', 'exit', None),
    }

    def run_all_figures():
        """Helper function to execute all figure generations sequentially."""
        print("\n--- RUNNING ALL FIGURES ---")
        for choice in sorted(menu_options.keys()):
            # Ensure 'run_all' and 'exit' are not called in the loop
            if menu_options[choice][1] not in ['run_all', 'exit']:
                execute_choice(choice)
        print("\n--- ALL FIGURES GENERATED ---")

    def execute_choice(choice):
        """Executes the function corresponding to the user's choice."""
        description, func, data_needed = menu_options[choice]

        if data_needed and not data_loaded:
            print(f"\nERROR: Cannot generate '{description}'. Market data failed to load.")
            return

        # Pass the correct dataset to the function
        if data_needed == 'full':
            func(full_data)
        elif data_needed == 'volatility':
            if volatility_data.empty:
                print("\nERROR: Volatility data slice is empty. Cannot generate VaR/GARCH plots.")
            else:
                func(volatility_data)
        else: # No data needed
            func()

    while True:
        print("\n" + "="*50)
        print("    REPRODUCIBLE ANALYSIS SCRIPT - MAIN MENU")
        print("="*50)
        for key, (desc, _, _) in menu_options.items():
            print(f"  [{key}] {desc}")
        print("-"*50)

        choice = input("Enter your choice: ").strip()

        if choice == '0':
            print("Exiting program.")
            break
        elif choice in menu_options:
            # Handle special commands like 'run_all' first
            if menu_options[choice][1] == 'run_all':
                run_all_figures()
            else:
                execute_choice(choice)
        else:
            print("Invalid choice. Please try again.")
        
        if choice in menu_options and choice != '0':
            input("\nPress Enter to return to the main menu...")


if __name__ == '__main__':
    effective_end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    print(f"Starting analysis for paper dated: {FINAL_ANALYSIS_DATE}")

    # --- Step 1: Load Data ---
    try:
        full_data = get_data(start_date=FULL_START_DATE, end_date=effective_end_date, cache_filename=CACHE_FILENAME)
        data_loaded = True
        print("Market data loaded successfully.")
    except (ConnectionError, ValueError) as e:
        print(f"\nCRITICAL ERROR: Could not load data. {e}")
        print("Data-driven visualizations will be unavailable.")
        full_data = pd.DataFrame()
        data_loaded = False

    # --- Step 2: Prepare Data Slices ---
    volatility_data = pd.DataFrame()
    if data_loaded and not full_data.empty:
        if pd.to_datetime(START_DATE_VOLATILITY) <= full_data.index.max():
            volatility_data = full_data.loc[START_DATE_VOLATILITY:]
        else:
            print("\nWARNING: Not enough data to create volatility slice. VaR/GARCH plots will be unavailable.")

    # --- Step 3: Launch Main Menu ---
    main_menu(full_data, volatility_data, data_loaded)
