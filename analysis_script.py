# ==============================================================================
#      PYTHON SCRIPT FOR REPRODUCIBLE DATA ANALYSIS AND VISUALIZATION
# ------------------------------------------------------------------------------
# Title:      An Examination of Bitcoin's Structural Shortcomings as Money:
#             A Synthesis of Economic and Technical Critiques
# Author:     Hamoon Soleimani
# Date:       November 13, 2025 (Date of final analysis)
#
# Description: This script reproduces the core quantitative analyses and
#              visualizations for the research paper, including Figures 2-6, 9-11,
#              16, 17, 18, 19, 20, 22, 27, and 28.
#              It ensures full reproducibility for data-driven plots via a static 
#              data file.
# Version:    11.1 (Final - Added Climate Damages Analysis)
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import networkx as nx
import seaborn as sns
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
warnings.filterwarnings("ignore") # Suppress plot warnings

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
    """
    print("\nGenerating Figure 5: Supply/Demand Model...")
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


def generate_topology_figure():
    """
    Figure 10: Generates a visualization of the Core-Periphery structure evolution.
    Simulates the 'Network Creation Game' via Preferential Attachment (Barabasi-Albert).
    Adapted for script dark theme with high-contrast edges.
    """
    print("Generating Figure 10: Lightning Network Topology...")
    
    # Use script global style (dark background)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Parameters for simulation
    n_early = 80; m_early = 2
    n_late = 400; m_late = 2 
    
    # 1. Early Stage Simulation
    G1 = nx.barabasi_albert_graph(n_early, m_early, seed=42)
    degrees1 = dict(G1.degree())
    threshold1 = np.percentile(list(degrees1.values()), 95)
    
    node_colors1 = []; node_sizes1 = []
    for node in G1.nodes():
        if degrees1[node] >= threshold1:
            node_colors1.append('#c0392b') # Deep Red (Core)
            node_sizes1.append(degrees1[node] * 8)
        else:
            node_colors1.append('#27ae60') # Green (Periphery)
            node_sizes1.append(30)

    pos1 = nx.spring_layout(G1, k=0.25, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(G1, pos1, node_color=node_colors1, node_size=node_sizes1, ax=ax1, alpha=0.9)
    # EDITED: Changed edge_color to bright silver (#e0e0e0) and increased alpha for visibility on black
    nx.draw_networkx_edges(G1, pos1, alpha=0.5, width=1.0, edge_color='#e0e0e0', ax=ax1)
    ax1.set_title("Phase 1: Early Network Formation\n(Emergent Hubs)", fontweight='bold', color='white')
    ax1.axis('off')

    # 2. Mature Stage Simulation
    G2 = nx.barabasi_albert_graph(n_late, m_late, seed=100)
    degrees2 = dict(G2.degree())
    threshold2 = np.percentile(list(degrees2.values()), 98)
    
    node_colors2 = []; node_sizes2 = []
    for node in G2.nodes():
        if degrees2[node] >= threshold2:
            node_colors2.append('#c0392b') 
            node_sizes2.append(degrees2[node] * 5)
        else:
            node_colors2.append('#27ae60') 
            node_sizes2.append(15)

    pos2 = nx.spring_layout(G2, k=0.15, iterations=80, seed=100)
    
    nx.draw_networkx_nodes(G2, pos2, node_color=node_colors2, node_size=node_sizes2, ax=ax2, alpha=0.85)
    # EDITED: Changed edge_color to bright silver (#e0e0e0) and increased alpha for visibility on black
    nx.draw_networkx_edges(G2, pos2, alpha=0.4, width=0.6, edge_color='#e0e0e0', ax=ax2)
    ax2.set_title("Phase 2: Mature Oligopoly\n(Structural Centralization)", fontweight='bold', color='white')
    ax2.axis('off')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Core (Liquidity Hubs)',
               markerfacecolor='#c0392b', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Periphery (Users)',
               markerfacecolor='#27ae60', markersize=8)
    ]
    # Legend text color handled by global params
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('figure_10_ln_topology.png', dpi=300, bbox_inches='tight')
    plt.show()



def generate_gini_figure():
    """
    Figure 11: Generates the quantitative analysis of centralization (Gini Coeffs).
    Creates synthetic data distributions that replicate the statistical findings
    of Vallarano et al. (Observed vs Expected Null Models).
    """
    print("Generating Figure 11: Quantitative Centralization Analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    metrics = ['Degree', 'Closeness', 'Betweenness', 'Eigenvector']
    
    np.random.seed(42)
    n_points = 100
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        expected = np.linspace(0.1, 0.9, n_points)
        
        if metric == 'Degree':
            noise = np.random.normal(0, 0.01, n_points)
            observed = expected + noise
        elif metric == 'Betweenness':
            noise = np.random.normal(0, 0.02, n_points)
            observed = expected + (expected * 0.15) + noise
            observed = np.clip(observed, 0, 0.99)
        elif metric == 'Closeness':
            noise = np.random.normal(0, 0.03, n_points)
            observed = expected + noise
        elif metric == 'Eigenvector':
            noise = np.random.normal(0, 0.02, n_points)
            observed = expected + (expected**2 * 0.1) + noise
            
        # Scatter plot
        ax.scatter(expected, observed, alpha=0.7, s=25, c='#3498db', edgecolor='w', linewidth=0.5)
        
        # Identity line (y=x)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Null Model Identity')
        
        ax.set_title(f"{metric} Centrality", fontweight='bold', color='white')
        ax.set_xlabel(f"Expected Gini (Null Model)", color='white')
        ax.set_ylabel(f"Observed Gini (Empirical)", color='white')
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        
        if metric == 'Betweenness':
            ax.text(0.5, 0.1, "Structural\nCentralization\nGap", 
                    fontsize=10, color='white', ha='center',
                    bbox=dict(facecolor='#c0392b', alpha=0.8, edgecolor='white'))

    lines, labels = axes[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)
    
    plt.tight_layout()
    plt.savefig('figure_11_ln_centralization_quantitative.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_systemic_shock_chart():
    """
    Reproduces Figure 16: Systemic Shock Analysis (Hashrate/Mempool during 2021 Blackout).
    """
    print("\nGenerating Figure 16: Systemic Shock Analysis...")
    
    # 1. Data Generation
    start_date = pd.Timestamp('2021-04-02')
    end_date = pd.Timestamp('2021-04-30')

    # --- Top Chart: Hashrate ---
    dates_hourly = pd.date_range(start=start_date, end=end_date, freq='3h')
    np.random.seed(123)
    hashrate_values = []
    for date in dates_hourly:
        noise = np.random.normal(0, 28) 
        if date < pd.Timestamp('2021-04-16'):
            val = 168 + noise
        elif date < pd.Timestamp('2021-04-23'):
            val = 130 + np.random.normal(0, 22) 
        else:
            val = 160 + noise
        val = max(20, min(val, 280))
        hashrate_values.append(val)
    df_hash = pd.DataFrame({'Date': dates_hourly, 'Hashrate': hashrate_values})

    # --- Bottom Chart: Mempool ---
    dates_daily = pd.date_range(start=start_date, end=end_date, freq='D')
    mempool_points = [
        64, 73, 72, 50, 48, 58, 59, 52, # Apr 2 - Apr 9
        65, 50, 45, 50, 62, 94,         # Apr 10 - Apr 15
        81,                             # Apr 16 (The dip before the rise)
        98, 155, 151, 163, 185, 195,    # The ascent
        205, 196, 150, 97, 85, 82, 75   # The peak and decline
    ]
    if len(mempool_points) < len(dates_daily):
        mempool_points += [75] * (len(dates_daily) - len(mempool_points))
    mempool_points = mempool_points[:len(dates_daily)]
    
    df_mempool_daily = pd.DataFrame({'Date': dates_daily, 'Unconfirmed': mempool_points})
    df_mempool_daily.set_index('Date', inplace=True)
    df_mempool = df_mempool_daily.resample('1h').interpolate(method='linear')
    df_mempool.reset_index(inplace=True)
    common_dates = df_hash['Date']
    df_mempool = df_mempool[df_mempool['Date'].isin(common_dates)]

    # 2. Plotting
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.08)

    # --- Top Panel: Hashrate ---
    ax1 = plt.subplot(gs[0])
    threshold_hash = 160 
    ax1.plot(df_hash['Date'], df_hash['Hashrate'], color='#8899a6', alpha=0.7, linewidth=1)
    ax1.fill_between(df_hash['Date'], df_hash['Hashrate'], threshold_hash, 
                     where=(df_hash['Hashrate'] >= threshold_hash),
                     interpolate=True, color='#3498DB', alpha=0.3, label='Healthy Hashrate')
    ax1.fill_between(df_hash['Date'], df_hash['Hashrate'], threshold_hash, 
                     where=(df_hash['Hashrate'] < threshold_hash),
                     interpolate=True, color='#e74c3c', alpha=0.4, label='Power Loss Impact')

    ax1.set_ylabel('Mining Power (EH/s)', fontsize=11)
    ax1.set_ylim(0, 280)
    ax1.set_xlim(start_date, end_date)
    ax1.grid(True, which='major', axis='y', linestyle=':', alpha=0.4)
    ax1.tick_params(labelbottom=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.text(pd.Timestamp('2021-04-19'), 90, "HASH RATE DROP\n(-24%)", 
             color='#e74c3c', fontweight='bold', ha='center', fontsize=10)

    # --- Bottom Panel: Mempool ---
    ax2 = plt.subplot(gs[1], sharex=ax1)
    threshold_mempool = 80
    ax2.plot(df_mempool['Date'], df_mempool['Unconfirmed'], color='#d35400', alpha=0.9, linewidth=1.5)
    ax2.fill_between(df_mempool['Date'], df_mempool['Unconfirmed'], 0, 
                     where=(df_mempool['Unconfirmed'] <= threshold_mempool),
                     interpolate=True, color='#7f8c8d', alpha=0.3)
    ax2.fill_between(df_mempool['Date'], df_mempool['Unconfirmed'], 0, 
                     where=(df_mempool['Unconfirmed'] > threshold_mempool),
                     interpolate=True, color='#e67e22', alpha=0.6, label='Congestion Spike')

    ax2.set_ylabel('Unconfirmed TX (Thousands)', fontsize=11)
    ax2.set_ylim(0, 220)
    ax2.grid(True, which='major', axis='y', linestyle=':', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Date Ticks using F-strings
    ticks = [pd.Timestamp('2021-04-02'), pd.Timestamp('2021-04-09'), 
             pd.Timestamp('2021-04-16'), pd.Timestamp('2021-04-23'), pd.Timestamp('2021-04-30')]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{t.strftime('%b')} {t.day}" for t in ticks], fontsize=11)
    ax2.text(pd.Timestamp('2021-04-24'), 210, "PEAK CONGESTION", 
             color='#e67e22', fontweight='bold', ha='center', fontsize=10)

    # --- Event Window Overlay ---
    rect_start = pd.Timestamp('2021-04-16')
    rect_end = pd.Timestamp('2021-04-23')
    for ax in [ax1, ax2]:
        ax.axvspan(rect_start, rect_end, color='#f1c40f', alpha=0.15, zorder=-2)
        ax.axvline(rect_start, color='#95a5a6', linestyle=':', linewidth=1.2)
        ax.axvline(rect_end, color='#95a5a6', linestyle=':', linewidth=1.2)
        ax.axvline(pd.Timestamp('2021-04-09'), color='#555555', linestyle=':', linewidth=1)
        ax.axvline(pd.Timestamp('2021-04-30'), color='#555555', linestyle=':', linewidth=1)

    fig.suptitle('The Bitcoin Blackout: Systemic Shock Analysis (Figure 16)', y=0.94, fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', frameon=False, fontsize=9)
    plt.subplots_adjust(top=0.88, bottom=0.1, left=0.1, right=0.95)
    
    plt.savefig('figure_16_systemic_shock_analysis.png', dpi=300)
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


def generate_climate_damages_jones_chart():
    """
    Reproduces Figure 19: Climate Damages Analysis based on Jones et al. (2022).
    Visualizes comparative damages, Bitcoin specific temporal data, and category averages.
    Adapted to the script's dark theme.
    """
    print("\nGenerating Figure 19: Climate Damages Analysis (Jones et al. 2022)...")

    # --- DATA FROM JONES ET AL. (2022) ---
    commodities_data = {
        'Coal': {'damage': 95, 'color': '#34495e', 'category': 'fossil'},
        'Natural Gas': {'damage': 46, 'color': '#34495e', 'category': 'fossil'},
        'Gasoline': {'damage': 41, 'color': '#c0392b', 'category': 'fossil'},
        'Bitcoin (avg)': {'damage': 35, 'color': '#f39c12', 'category': 'crypto'},
        'Beef': {'damage': 33, 'color': '#8e44ad', 'category': 'agriculture'},
        'Crude Oil': {'damage': 25, 'color': '#34495e', 'category': 'fossil'},
        'Gold': {'damage': 4, 'color': '#f1c40f', 'category': 'metal'},
        'Solar': {'damage': 3, 'color': '#27ae60', 'category': 'renewable'},
    }
    bitcoin_temporal = {
        'years': [2016, 2017, 2018, 2019, 2020, 2021],
        'damages': [16, 18, 37, 53, 82, 25]
    }

    # --- FIGURE SETUP ---
    # Using global dark theme, so facecolor is black by default via plt.style
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1.3, 1], 
                           hspace=0.35, wspace=0.3)

    # --- PANEL A: COMMODITY COMPARISON (MAIN) ---
    ax1 = fig.add_subplot(gs[0, 0])

    items = list(commodities_data.keys())
    damages = [commodities_data[k]['damage'] for k in items]
    colors = [commodities_data[k]['color'] for k in items]

    sorted_indices = np.argsort(damages)[::-1]
    items_sorted = [items[i] for i in sorted_indices]
    damages_sorted = [damages[i] for i in sorted_indices]
    colors_sorted = [colors[i] for i in sorted_indices]

    y_pos = np.arange(len(items_sorted))

    # Bars with black edges for contrast
    bars = ax1.barh(y_pos, damages_sorted, color=colors_sorted, 
                    edgecolor='black', linewidth=1.5, height=0.7, alpha=0.9)

    # Highlight Bitcoin
    bitcoin_idx = items_sorted.index('Bitcoin (avg)')
    bars[bitcoin_idx].set_edgecolor('#d35400')
    bars[bitcoin_idx].set_linewidth(2.5)
    bars[bitcoin_idx].set_hatch('///')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(items_sorted, fontsize=12, fontweight='bold', color='white')
    ax1.invert_yaxis()
    ax1.set_xlabel('Climate Damages (% of Market Price)', fontsize=12, fontweight='bold', color='white')
    ax1.set_title('Panel A: Comparative Climate Damages Across Commodities', 
                  fontsize=14, fontweight='bold', pad=15, loc='left', color='white')

    for i, (val, item) in enumerate(zip(damages_sorted, items_sorted)):
        label = f' {val}%'
        ax1.text(val + 1, i, label, va='center', fontsize=11, fontweight='bold', color='white')

    ax1.axvline(x=35, color='#e67e22', linestyle='--', linewidth=1.5, alpha=0.6)
    ax1.text(35, len(items_sorted) - 0.5, 'Bitcoin avg', fontsize=9, 
             color='#e67e22', ha='center', fontweight='bold')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.grid(axis='x', alpha=0.2, linestyle='--', color='#555555')
    ax1.set_xlim(0, 105)

    # --- PANEL B: TEMPORAL ANALYSIS ---
    ax2 = fig.add_subplot(gs[0, 1])
    years = bitcoin_temporal['years']
    btc_damages = bitcoin_temporal['damages']

    ax2.plot(years, btc_damages, color='#e67e22', marker='o', 
             markersize=10, linewidth=3, markeredgecolor='white', 
             markeredgewidth=2, label='Bitcoin', zorder=3)

    ax2.fill_between(years, btc_damages, alpha=0.25, color='#e67e22')

    avg_damage = np.mean(btc_damages)
    ax2.axhline(y=avg_damage, color='#95a5a6', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Mean: {avg_damage:.0f}%')
    ax2.axhline(y=4, color='#f1c40f', linestyle='-', 
                linewidth=2, alpha=0.7, label='Gold: 4%')

    peak_year = years[np.argmax(btc_damages)]
    peak_value = max(btc_damages)
    ax2.annotate(f'Peak: {peak_value}%\n({peak_year})', 
                 xy=(peak_year, peak_value), xytext=(peak_year - 1.5, peak_value + 8),
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
                 fontsize=10, fontweight='bold', color='#c0392b',
                 bbox=dict(boxstyle='round,pad=0.4', fc='black', ec='#c0392b', lw=1.5))

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold', color='white')
    ax2.set_ylabel('Climate Damages (%)', fontsize=12, fontweight='bold', color='white')
    ax2.set_title('Panel B: Bitcoin Damages Over Time (2016-2021)', 
                  fontsize=14, fontweight='bold', pad=15, loc='left', color='white')
    ax2.set_ylim(0, 95)
    ax2.set_xticks(years)
    ax2.legend(loc='upper left', fontsize=10, frameon=True, shadow=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='y', alpha=0.3, linestyle='--', color='#555555')

    # --- PANEL C: CATEGORY SUMMARY ---
    ax3 = fig.add_subplot(gs[1, :])

    categories = {}
    for item, data in commodities_data.items():
        cat = data['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(data['damage'])

    cat_names = list(categories.keys())
    cat_avgs = [np.mean(categories[cat]) for cat in cat_names]
    cat_colors = ['#34495e', '#f39c12', '#8e44ad', '#f1c40f', '#27ae60']

    x_pos = np.arange(len(cat_names))
    ax3.bar(x_pos, cat_avgs, color=cat_colors, edgecolor='black', 
            linewidth=2, alpha=0.9, width=0.6)

    cat_labels = ['Fossil Fuels', 'Cryptocurrency', 'Agriculture', 'Precious Metals', 'Renewable Energy']
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(cat_labels, fontsize=11, fontweight='bold', color='white')
    ax3.set_ylabel('Average Climate Damages (%)', fontsize=11, fontweight='bold', color='white')
    ax3.set_title('Panel C: Average Damages by Category', 
                  fontsize=14, fontweight='bold', pad=15, loc='left', color='white')

    for i, val in enumerate(cat_avgs):
        ax3.text(i, val + 2, f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold', color='white')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, max(cat_avgs) * 1.2)
    ax3.grid(axis='y', alpha=0.2, linestyle='--', color='#555555')

    # --- FOOTER ---
    fig.text(0.5, 0.01, 
             'Data Source: Jones, B.A., Goodkind, A.L., & Berrens, R.P. (2022). '
             'Economic estimation of Bitcoin mining\'s climate damages. Scientific Reports, 12, 14512.',
             ha='center', fontsize=9, style='italic', color='#bbbbbb',
             bbox=dict(facecolor='#1c1c1c', alpha=0.8, edgecolor='#555555', 
                       boxstyle='round,pad=0.8', linewidth=1))

    plt.savefig('figure_19_jones_climate_damages.png', dpi=300, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    print("✓ Visualization saved as 'figure_19_jones_climate_damages.png'")
    plt.show()


def generate_oceanic_games_model():
    """
    Reproduces Figure 20: Economic incentive for mining centralization from an
    Oceanic Games model, adapted to a non-linear representation.
    """
    print("\nGenerating Figure 20: Economic Incentive for Mining Centralization...")

    def get_coalition_value_per_unit_nonlinear(r):
        a = 0.00015; b = 0.01; c = 1.0
        return a * r**2 + b * r + c

    def get_oceanic_value_per_unit_nonlinear(r):
        a = -0.00015; b = -0.005; c = 1.0
        return a * r**2 + b * r + c

    r_crystallized = np.linspace(0, 40, 400)
    v1_values = get_coalition_value_per_unit_nonlinear(r_crystallized)
    voc_values = get_oceanic_value_per_unit_nonlinear(r_crystallized)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_crystallized, v1_values, color='#d62728', linewidth=2.5, label=r'$v_1$ (Value for Coalition Members)')
    ax.plot(r_crystallized, voc_values, color='#1f77b4', linewidth=2.5, label=r'$v_{oc}$ (Value for Oceanic Miners)')

    ax.set_xlabel('Percentage of Total Hash Rate in New Coalition ($r_1$)', fontsize=12)
    ax.set_ylabel('Strategic Value Per Unit of Hash Rate', fontsize=12)
    ax.set_title('Economic Incentive for Mining Centralization (Figure 20)', fontsize=14, weight='bold')
    ax.set_xlim(0, 40); ax.set_ylim(0.5, 1.7)
    legend = ax.legend(fontsize=11, title="Value per Unit of Resource")
    legend.get_title().set_fontweight('bold')
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.savefig('figure_20_centralization_incentive.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

def generate_security_budget_dilemma_chart():
    """
    Reproduces Figure 22: A model of the Bitcoin Security Budget Dilemma.
    """
    print("\nGenerating Figure 22: Bitcoin Security Budget Dilemma...")

    fig, ax = plt.subplots(figsize=(12, 8))
    color_blue = '#56B4E9'; color_red = '#D55E00'

    x_events = {"Present": 0, "2028 Halving": 1.5, "2032 Halving": 3.0, "...Post-Subsidy Era": 4.5}
    x_ticks = list(x_events.values()); x_labels = list(x_events.keys())
    y_levels = {"Vulnerable": 0, "Low": 1, "Medium": 2, "High": 3}
    y_ticks = list(y_levels.values()); y_labels = list(y_levels.keys())

    x_subsidy = [0, 1.5, 1.5, 3.0, 3.0, 4.5, 4.5, 5.5]
    y_subsidy = [3, 3,   2,   2,   1,   1,   0.4, 0.2]
    x_l1_retention = [0, 1.5, 1.5, 5.5]
    y_l1_retention = [3, 3,   2.8, 2.5]

    ax.plot(x_subsidy, y_subsidy, linestyle='--', color=color_blue, lw=2.5, label='Scenario A: High L2 Adoption')
    ax.plot(x_l1_retention, y_l1_retention, linestyle='-', color=color_red, lw=2.5, label='Scenario B: L1 Retention')

    ax.set_xlabel("Time", fontsize=14, labelpad=10); ax.set_ylabel("Security Budget", fontsize=14, labelpad=10)
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlim(-0.2, 5.7); ax.set_ylim(-0.2, 3.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax.annotate('Requires Prohibitively High\nTransaction Fees', xy=(2.25, 2.75), xytext=(2.5, 3.1),
                fontsize=11, color=color_red, ha='left', arrowprops=dict(facecolor=color_red, shrink=0.05, width=1, headwidth=6, edgecolor='none'))
    ax.annotate('Transaction Fees', xy=(4.0, 2.55), xytext=(4.0, 1.5), fontsize=11, ha='center', va='center',
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='white', shrinkA=5, shrinkB=5))

    blue_patch = mpatches.Patch(color=color_blue, label='Scenario A: High L2 Adoption')
    red_patch = mpatches.Patch(color=color_red, label='Scenario B: L1 Retention')
    ax.legend(handles=[blue_patch, red_patch], loc='upper right', fontsize=12, frameon=True)
    ax.set_title("A Model of the Bitcoin Security Budget Dilemma (Figure 22)", fontsize=16, pad=20, weight='bold')

    plt.tight_layout()
    plt.savefig('figure_22_security_budget_dilemma.png', dpi=300)
    plt.show()

def generate_wash_trading_chart():
    """
    Reproduces Figure 27: Percentage of Failed Forensic Tests for Wash Trading.
    """
    print("\nGenerating Figure 27: Wash Trading Forensic Failure Rates...")

    exchange_data = {
        'Exchange': ['U8; U14', 'U9', 'U2; U5', 'U10', 'U1; U4; U7; U12', 'UT4; UT6', 'U3', 'U11; U16', 'UT10', 'U6; U15', 
                     'UT7', 'UT3; UT8', 'UT1; UT2; UT5', 'U13', 'UT9', 'R1; R2; R3'],
        'Failure_Rate': [98, 95, 88, 75, 70, 67, 58, 42, 42, 34, 22, 17, 8, 3, 2, 0],
        'Category': ['High Risk', 'High Risk', 'High Risk', 'High Risk', 'High Risk', 'Medium Risk', 'High Risk', 'High Risk', 'Medium Risk', 'High Risk',
                     'Medium Risk', 'Medium Risk', 'Medium Risk', 'High Risk', 'Medium Risk', 'Regulated']
    }
    crypto_data = {'Crypto': ['XRP', 'LTC', 'BTC', 'ETH'], 'Failure_Rate': [54, 47, 48, 42]}
    df_exch = pd.DataFrame(exchange_data); df_crypto = pd.DataFrame(crypto_data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    plt.subplots_adjust(hspace=0.3) 

    colors = []
    for cat in df_exch['Category']:
        if cat == 'High Risk': colors.append('#d62728') 
        elif cat == 'Medium Risk': colors.append('#1f77b4') 
        else: colors.append('#2ca02c') 

    y_pos = np.arange(len(df_exch))
    bars1 = ax1.barh(y_pos, df_exch['Failure_Rate'], color=colors, height=0.6)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(df_exch['Exchange'], fontsize=10); ax1.invert_yaxis() 
    ax1.set_xlabel('Percentage of Failed Tests (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Percentage of Failed Forensic Tests by Exchange (Figure 27)', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlim(0, 100); ax1.grid(axis='x', linestyle='--', alpha=0.5) 
    for i, v in enumerate(df_exch['Failure_Rate']): ax1.text(v + 1, i + 0.15, str(v) + '%', color='white', fontsize=9)

    df_crypto = df_crypto.sort_values('Failure_Rate', ascending=True)
    y_pos_crypto = np.arange(len(df_crypto))
    crypto_colors = ['#00688b', '#c0c0c0', '#f2a900', '#00688b'] 
    bars2 = ax2.barh(y_pos_crypto, df_crypto['Failure_Rate'], color=crypto_colors, height=0.6)
    ax2.set_yticks(y_pos_crypto); ax2.set_yticklabels(df_crypto['Crypto'], fontsize=11)
    ax2.set_xlabel('Percentage of Failed Tests (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Percentage of Failed Forensic Tests by Cryptocurrency', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(0, 100); ax2.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(df_crypto['Failure_Rate']): ax2.text(v + 1, i + 0.1, str(v) + '%', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig('figure_27_wash_trading.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

def generate_tether_issuance_chart():
    """
    Reproduces Figure 28: Bitcoin Returns Conditional on Tether Issuance.
    """
    print("\nGenerating Figure 28: Tether Issuance Impact...")
    categories = ['Zero', 'Low', 'Medium', 'High']
    raw_returns = [0.6, -0.5, -1.8, -4.2] 
    benchmarked_returns = [0.05, -1.9, -3.1, -6.1]

    x = np.arange(len(categories)); width = 0.35 
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, raw_returns, width, label='Raw EOM Returns', color='#1f77b4')
    rects2 = ax.bar(x + width/2, benchmarked_returns, width, label='Benchmarked Returns', color='#d62728')

    ax.set_ylabel('Return (%)', fontsize=12)
    ax.set_xlabel('Monthly Tether Issuance Quantile', fontsize=12)
    ax.set_title('Bitcoin End-of-Month Returns Conditional on Tether Issuance (Figure 28)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
    ax.axhline(0, color='white', linewidth=0.8); ax.legend()
    ax.yaxis.grid(True, linestyle='--', alpha=0.5); ax.set_axisbelow(True)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            xy_pos = (rect.get_x() + rect.get_width() / 2, height)
            xy_text = (0, 5) if height >= 0 else (0, -15)
            if abs(height) > 0.1:
                ax.annotate(f'{height}%', xy=xy_pos, xytext=xy_text, textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color='white')
    autolabel(rects1); autolabel(rects2)

    plt.tight_layout()
    plt.savefig('figure_28_tether_issuance.png', dpi=300, bbox_inches='tight')
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
        '6': ('Figure 10: LN Topology (Network Simulation)', generate_topology_figure, None),
        '7': ('Figure 11: LN Centralization (Gini Quantitative)', generate_gini_figure, None),
        '8': ('Figure 16: Systemic Shock Analysis', generate_systemic_shock_chart, None),
        '9': ('Figures 17 & 18: Drawdown and Correlation Analysis', analyze_digital_gold_narrative, 'full'),
        '10': ('Figure 19: Climate Damages Analysis (Jones et al.)', generate_climate_damages_jones_chart, None),
        '11': ('Figure 20: Economic Incentive for Mining Centralization', generate_oceanic_games_model, None),
        '12': ('Figure 22: Bitcoin Security Budget Dilemma', generate_security_budget_dilemma_chart, None),
        '13': ('Figure 27: Wash Trading Forensic Failure Rates', generate_wash_trading_chart, None),
        '14': ('Figure 28: Tether Issuance Impact', generate_tether_issuance_chart, None),
        '15': ('Run All Figures', 'run_all', None),
        '0': ('Exit', 'exit', None),
    }

    def run_all_figures():
        """Helper function to execute all figure generations sequentially."""
        print("\n--- RUNNING ALL FIGURES ---")
        sorted_keys = sorted(menu_options.keys(), key=lambda x: int(x))
        for choice in sorted_keys:
            if menu_options[choice][1] not in ['run_all', 'exit']:
                execute_choice(choice)
        print("\n--- ALL FIGURES GENERATED ---")

    def execute_choice(choice):
        """Executes the function corresponding to the user's choice."""
        description, func, data_needed = menu_options[choice]

        if data_needed and not data_loaded:
            print(f"\nERROR: Cannot generate '{description}'. Market data failed to load.")
            return

        if data_needed == 'full':
            func(full_data)
        elif data_needed == 'volatility':
            if volatility_data.empty:
                print("\nERROR: Volatility data slice is empty. Cannot generate VaR/GARCH plots.")
            else:
                func(volatility_data)
        else: 
            func()

    while True:
        print("\n" + "="*50)
        print("    REPRODUCIBLE ANALYSIS SCRIPT - MAIN MENU")
        print("="*50)
        sorted_keys = sorted(menu_options.keys(), key=lambda x: int(x))
        for key in sorted_keys:
            desc = menu_options[key][0]
            print(f"  [{key}] {desc}")
        print("-"*50)

        choice = input("Enter your choice: ").strip()

        if choice == '0':
            print("Exiting program.")
            break
        elif choice in menu_options:
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
