# Bitcoin's Structural Position as Money: A Contested Synthesis

**Author:** Hamoon Soleimani  
**Date:** September 2026  

This repository contains the LaTeX source code and the fully reproducible Python analysis script for the paper **"Bitcoin's Structural Position as Money: A Contested Synthesis of Post-Keynesian and Austrian Critiques."**

## Overview

The paper evaluates Bitcoin's viability as a monetary standard against two competing theoretical traditions (Post-Keynesian monetary theory and the Austrian School) and tests these frameworks against current empirical data. 

Key empirical analyses include:
*   **Econometric Volatility & Value-at-Risk (VaR):** Out-of-sample backtested Extreme Value Theory (EVT) and Cornish-Fisher VaR measurements against equities, gold, and fiat. 
*   **The Settlement-Throughput Trilemma:** Comparing Bitcoin L1 to Sovereign RTGS systems (Fedwire, T2) rather than retail card networks.
*   **Lightning Network Centralization:** Reconciling the 2020 decentralized mesh failure rates with 2023–2026 hub-and-spoke success rates, confirming predicted game-theoretic centralization.
*   **51% Attack Economics & Mining Concentration:** Analysis of modern attack vectors (including derivatives-funded exploits) and hashrate concentration.
*   **Macro-Liquidity Confounders:** Partial-correlation tests isolating Bitcoin's relationship to equities net of DXY and VIX co-movements.
*   **El Salvador Natural Experiment:** Difference-in-differences reproduction of the macroeconomic impacts of Bitcoin as legal tender.

## Repository Structure

*   `analysis_script.py`: The reproducible Python script used to fetch current market data, execute all statistical tests, and generate the 19 figures and data tables found in the paper.
*   `/paper_outputs/`: Directory generated automatically by the Python script containing all exported `.pdf` / `.png` figures and `.csv` / `.tex` tables.

## Reproducing the Analysis

The provided Python script (`analysis_script.py`) allows anyone to perfectly recreate the quantitative models, figures, and data tables directly from live market data (up to the current date).

### Prerequisites

Ensure you have Python 3 installed. Install the required dependencies:

```bash
pip install yfinance arch numpy pandas matplotlib scipy statsmodels
```

### Running the Script

Execute the script via the command line:

```bash
python analysis_script.py
```

Upon execution, the script will:
1. Attempt to load cached historical data or fetch fresh data via `yfinance`.
2. Open an interactive CLI menu.
3. Allow you to generate individual figures/tables by entering the corresponding number, or press **19** to run all analyses at once.
4. Output all generated assets to the `paper_outputs/` directory.

## License
[MIT License](LICENSE) - Free for academic and personal use. Please cite the author if referencing the models or text.
