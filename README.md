# Bitcoin Structural Shortcomings Analysis

**Repository for the code and data supporting the research paper:**

*An Examination of Bitcoin's Structural Shortcomings as Money: A Synthesis of Economic and Technical Critiques*

**Author:** Hamoon Soleimani  
**Date of Analysis:** November 13, 2025

This repository provides the complete, open-source Python script and the static dataset required to reproduce the key quantitative findings, visualizations, and econometric models presented in the paper.

---

## 1. Reproducibility Guarantee

The goal of this repository is to ensure **100% reproducibility** of the data-driven figures (Figures 2, 3, 4, 5, 11, and 12).

*   **Static Data:** The analysis relies on historical price data for BTC-USD, GC=F, ^GSPC, UUP, and AAPL, finalized on **November 13, 2025**. This data is included in the file `research_data_static.csv`.
*   **Static Dates:** The analysis script is hardcoded to use the exact start and end dates used in the paper, ensuring that rerunning the script yields the identical results presented in the publication.

## 2. Repository Contents

| File | Description | Purpose |
| :--- | :--- | :--- |
| `analysis_script.py` | The main Python script implementing all time-series analysis, VaR calculations, and GARCH modeling. | Executes all quantitative figures. |
| `research_data_static.csv` | **Static dataset** containing the historical 'Close' prices for all tickers used in the analysis. | Ensures reproducibility independent of API changes. |
| `requirements.txt` | A list of all Python packages and their versions required to run the script. | Simplifies environment setup. |
| `README.md` | This instruction file. | Provides documentation and instructions. |
| `figure_*.png` (Output) | Generated PNG files (e.g., `figure_4_garch_volatility.png`). | Output visualization files (will be generated upon first run). |

## 3. Setup and Execution

### Prerequisites

You must have [Python 3](https://www.python.org/downloads/) installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YourUsername/Bitcoin_Structural_Shortcomings_Analysis.git
cd Bitcoin_Structural_Shortcomings_Analysis
```

### Step 2: Install Dependencies

Use the included `requirements.txt` file to install the required libraries (pandas, numpy, yfinance, arch, etc.):

```bash
pip install -r requirements.txt
```

### Step 3: Run the Analysis

Execute the main script. It will automatically load data from the static CSV file and generate all quantitative figures in PNG format within the current directory.

```bash
python analysis_script.py
```

## 4. Analysis Overview

The `analysis_script.py` performs the following key functions, referencing the figures in the paper:

| Function in Code | Analysis Performed | Figure | Citation/Reference |
| :--- | :--- | :--- | :--- |
| `generate_volatility_comparison_chart` | 15-day and 200-day Annualized Rolling Volatility for BTC, Gold, and Apple (log scale). | Figure 2 | Yermack (2014) |
| `analyze_risk_and_garch` | Calculates 1-Day 95% Value-at-Risk (VaR) for core assets. | Figure 3 | |
| `analyze_risk_and_garch` | Fits the GARCH(1,1) model to Bitcoin returns, calculating conditional volatility and half-life persistence. | Figure 4 | Chinazzo & Jeleskovic (2024) |
| `generate_tps_chart` | Visualization of Transaction Per Second (TPS) comparison against Visa and Mastercard. | Figure 5 | Visa 10-K (2024), Mastercard 10-K (2024) |
| `analyze_digital_gold_narrative` | Calculates and visualizes Bitcoin's maximum historical drawdown from All-Time Highs. | Figure 11 | iShares (2025) |
| `analyze_digital_gold_narrative` | Calculates and plots the 60-Day Rolling Correlation between Bitcoin and the S&P 500. | Figure 12 | Conlon et al. (2020) |

