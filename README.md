# Bitcoin Structural Shortcomings Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repository for the code and data supporting the research paper:

**"An Examination of Bitcoin's Structural Shortcomings as Money: A Synthesis of Economic and Technical Critiques"**

**Author:** Hamoon Soleimani  
**Analysis Date:** November 13, 2025  
**Script Version:** 11.2

---

## 📋 Overview

This repository provides a complete, open-source Python implementation for reproducing all quantitative findings, visualizations, and econometric models presented in the research paper. The analysis examines Bitcoin's structural limitations across multiple dimensions including volatility, network topology, economic incentives, and environmental impact.

---

## 🎯 Reproducibility Guarantee

Full reproducibility is ensured through:

- **Static Dataset**: Analysis uses `research_data_static.csv` containing historical price data for BTC-USD, Gold (GC=F), S&P 500 (^GSPC), USD Index (UUP), and Apple (AAPL), finalized on November 13, 2025
- **API Fallback**: Automatic data fetching via `yfinance` API if static file is unavailable
- **Hardcoded Parameters**: Critical analysis dates (e.g., Drawdown start: 2015-01-01) are fixed to match published results

---

## 📁 Repository Structure

```
Bitcoin_Structural_Shortcomings_Analysis/
├── analysis_script.py              # Main Python script with all analysis logic
├── research_data_static.csv        # Static historical price dataset
├── requirements.txt                # Python package dependencies
├── README.md                       # This file
└── figures/                        # Generated visualizations (auto-created)
    ├── figure_2.png
    ├── figure_3.png
    └── ...
```

### File Descriptions

| File | Purpose |
|------|---------|
| `analysis_script.py` | Contains all analysis logic, simulation models, and plotting functions |
| `research_data_static.csv` | Historical 'Close' prices for reproducibility independent of API changes |
| `requirements.txt` | Complete list of required Python packages for easy environment setup |
| `figure_*.png` | Output visualizations generated automatically by the script |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/Bitcoin_Structural_Shortcomings_Analysis.git
   cd Bitcoin_Structural_Shortcomings_Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key packages include: `pandas`, `yfinance`, `arch`, `networkx`, `matplotlib`, `seaborn`, `scipy`, `numpy`

3. **Run the analysis**
   ```bash
   python analysis_script.py
   ```

---

## 📊 Interactive Menu System

The script features an interactive menu allowing you to generate specific figures or run the complete analysis suite:

### Available Options

1. **Figure 2** - Comparative Rolling Volatility (BTC, Gold, Apple)
2. **Figures 3 & 4** - Value-at-Risk (VaR) & GARCH(1,1) Volatility Modeling
3. **Figure 5** - Supply/Demand Model (Theoretical Fixed vs. Elastic Supply)
4. **Figure 6** - TPS Capacity Comparison (Log Scale)
5. **Figure 9** - Lightning Network Centralization Parameter Map
6. **Figure 10** - Lightning Network Topology Simulation (Barabási-Albert)
7. **Figure 11** - Quantitative Centralization Analysis (Gini Coefficients)
8. **Figure 16** - Systemic Shock Analysis (2021 Hashrate/Mempool Blackout)
9. **Figures 17 & 18** - Drawdown Analysis & Correlation with S&P 500
10. **Figure 19** - Climate Damages Analysis (Based on Jones et al., 2022)
11. **Figure 20** - Economic Incentive for Mining Centralization (Oceanic Games)
12. **Figure 22** - Bitcoin Security Budget Dilemma Model
13. **Figure 27** - Wash Trading Forensic Failure Rates
14. **Figure 28** - Bitcoin Returns Conditional on Tether Issuance
15. **Figure 25** - Entity Distribution Analysis (Schnoering & Vazirgiannis)
16. **Run All Figures** - Generate complete analysis suite sequentially

All figures are generated using a custom dark theme for high contrast and saved as 300 DPI PNG files.

---

## 🔬 Analysis Methodology

### Analytical Components

| Analysis Type | Methodology | Figures |
|---------------|-------------|---------|
| **Volatility Analysis** | Annualized rolling standard deviation, GARCH(1,1) with Student's t-distribution, 95% VaR | 2, 3, 4 |
| **Network Topology** | NetworkX simulations, Barabási-Albert preferential attachment, Gini coefficient analysis | 9, 10, 11 |
| **Macroeconomic Models** | Supply curve elasticity modeling, comparative TPS analysis (Visa/Mastercard benchmarks) | 5, 6 |
| **On-Chain Forensics** | Entity clustering distribution, wash trading failure rates, Tether issuance correlation | 25, 27, 28 |
| **Systemic Risk** | Event study of April 2021 Xinjiang blackout (hashrate vs mempool congestion) | 16 |
| **Environmental Impact** | Comparative climate damages as percentage of market price | 19 |
| **Game Theory** | Economic incentives for mining centralization (Oceanic games), security budget modeling | 20, 22 |

### Key Features

- **Econometric Models**: GARCH volatility forecasting with conditional heteroskedasticity
- **Network Science**: Complex network analysis using preferential attachment models
- **Time Series Analysis**: Rolling correlations, drawdown calculations, and event studies
- **Comparative Analysis**: Multi-asset benchmarking against traditional financial instruments

---

## 📈 Output Specifications

- **Format**: PNG images at 300 DPI resolution
- **Theme**: Custom dark theme optimized for academic publications
- **Location**: Figures saved to working directory or `figures/` subfolder
- **Naming**: Consistent with paper figure numbers (e.g., `figure_2.png`)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{soleimani2025bitcoin,
  title={An Examination of Bitcoin's Structural Shortcomings as Money: A Synthesis of Economic and Technical Critiques},
  author={Soleimani, Hamoon},
  year={2025},
  month={November}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

---

**Last Updated:** November 13, 2025
