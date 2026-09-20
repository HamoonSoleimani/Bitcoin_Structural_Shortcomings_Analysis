# The Re-Intermediation Trilemma: Scaling, Liquidity, and Structural Centralization in Bitcoin

**Author:** Hamoon Soleimani  
**Reference Analysis Date:** September 5, 2026  
**Repository Version:** v7.1 (Manuscript Realignment & Rigorous Reproducibility Pass)

---

## Abstract & Theoretical Framework

This repository provides the complete empirical replication pipeline and econometric toolchain for the paper **"The Re-Intermediation Trilemma: Scaling, Liquidity, and Structural Centralization in Bitcoin."**

### The Core Thesis: Structural Re-Intermediation
The manuscript departs from broad, diffuse critiques of cryptocurrency to evaluate a single, testable thesis: **economies of scale in liquidity provision and coordination costs systematically reassert custodial intermediation at every layer of the Bitcoin architecture**, counteracting protocol-level disintermediation guarantees (Diamond, 1984; Katz & Shapiro, 1985).

The paper formalizes this through the **Re-Intermediation Trilemma** across three interconnected empirical pillars:

```
                        [ The Re-Intermediation Trilemma ]
                 Same economic mechanism driving custodial gravity:
             Scale economies in liquidity provision & coordination costs
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [ Pillar I: L1 ]              [ Pillar II: L2 ]             [ Pillar III: L0 ]
Settlement Finality             Routing Topology               Block Validation
  (Throughput Ceiling)           (Channel Liquidity)            (Coalition Values)
         │                              │                              │
         ▼                              ▼                              ▼
Wholesale RTGS-parity          Hub-and-spoke capital          Mining pool hashrate
(Fedwire/T2); retail-          concentration (Gini:           duopoly (43–57% top 2;
inadequate without L2          0.86 → 0.97, 2018–2025)        Stratum V2 prod: 3–5%)
```

1. **Pillar I — Settlement Layer (§3, Fig. 2):**  
   Bitcoin L1 throughput (~6.5 TPS) is quantitatively evaluated against sovereign Real-Time Gross Settlement (RTGS) systems (Fedwire at ~9.68 TPS via 2024 PFMI disclosures; TARGET2 at 400k–450k daily transfers). This confirms L1 is throughput-adequate for wholesale gross settlement, but mathematically incapable of retail transaction volume without secondary layers.
2. **Pillar II — Routing Layer (§4, Figs. 3–4):**  
   Game-theoretic routing models (Avarikioti et al., 2020) predict that channel capital opportunity costs force network topology to shift from a decentralized peer-to-peer mesh to capital-concentrated custodial hubs. Empirical evidence confirms this: decentralized active probes demonstrate steep failure rates for multi-dollar transfers (Waugh & Holz, 2020), while well-capitalized custodial nodes achieve 99.7% reliability (River Financial, 2023). Convergent multi-study graph metrics show channel-capacity Gini coefficients climbing continuously from 0.86 (2018) to 0.955 (2023) and 0.97 (2025).
3. **Pillar III — Validation Layer (§5, Figs. 5–6, Appendix C):**  
   Super-additive coalition values drive mining pool concentration (Leonardos et al., 2019). The top two pools control 43–57% of network hashrate, and the top 4–5 control 65–76%. Stratum V2 mitigates block-template censorship, but production deployment remains low (~3–5% as of mid-2026 vs. ~75% working-group commitment), leaving reorg risk and physical hashrate centralization unaddressed under declining block-reward security budgets (Budish, 2018).

---

## Manuscript Scope & v7 Restructuring Changelog

The repository codebase (`bitcoin_analysis_v7.py`) reflects a manuscript-level scope realignment:

* **Separation of Scope:** Peripheral, non-core discussions (the Post-Keynesian money hierarchy schematic, $MV=PY$ debt-deflation models, governance deadlocks, Basel III risk weights, and secondary regressions of third-party El Salvador datasets) have been excised from the main argument.
* **Preservation of Methodological Context:**
  * **Appendix A:** Econometric characterization of Bitcoin’s volatility dynamics, tail-risk metrics, and macro-liquidity correlations.
  * **Appendix B:** Market microstructure robustness checks (wash-trading volume shares across Bitwise, Cong et al., and Sila et al.; Tether/USDT stablecoin market-cap dominance).
  * **Appendix C:** Supplementary 51% attack-cost economics (Harvey’s static $6B hardware model updated to $8B with derivatives shorting).
* **Machine-Checkable Evidentiary Audit:** Every generated figure is cataloged in `table_figure_manifest.csv` with its formal evidentiary status (`empirical`, `reproduced`, `theoretical-model`, or `conceptual`) and manuscript section role.

---

## Econometric Methodology & Numerical Corrections

The pipeline enforces several econometric standards to prevent common modeling artifacts:

1. **Native Trading Calendars (No Premature Intersections):**  
   Assets are tracked on their native trading schedules (365 calendar days for Bitcoin; 252 trading days for equities, gold, and fiat). Downstream annualizations use $\sqrt{365}$ and $\sqrt{252}$ respectively. Date intersections occur strictly and solely via `build_aligned_panel` when paired observations are mathematically required.
2. **Strict Out-of-Sample VaR Backtesting:**  
   The rolling Value-at-Risk backtester trains on $t - \text{window} : t - 1$ and evaluates hits at realized time $t$, preserving the validity of Kupiec (POF) and Christoffersen (Independence) likelihood-ratio statistics.
3. **Apples-to-Apples Dynamic Conditional Correlation (DCC-GARCH):**  
   Comparisons of Bitcoin’s co-movement with the S&P 500 evaluate identical two-step DCC-GARCH(1,1) specifications (Engle, 2002) across raw returns and static OLS-residualized returns (orthogonalized against the Dollar Index and VIX). Optimization bounds enforce genuine stationarity ($a + b < 0.9999$) without artificial parameter ceiling constraints.
4. **Model-Consistent Volatility Persistence & Half-Life:**  
   Univariate volatility models are selected via Bayesian Information Criterion (BIC) across GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1) specs under Normal, Student's $t$, and Skewed $t$ densities. Shock persistence and half-life formulas adapt to the selected model structure (e.g., autoregressive parameter $\beta$ for EGARCH; $\alpha + \beta + 0.5\gamma$ for GJR-GARCH).
5. **Fixed Evaluation Cutoff:**  
   The pipeline enforces a fixed termination date (`2026-09-05`), ensuring identical replication regardless of when the script is executed.

---

## Repository Structure

```
.
├── bitcoin_analysis_v7.py              # Primary reproducible analysis engine
├── research_data_native_calendar_v7.csv# Cached native-calendar asset returns
├── paper_outputs/                      # Generated vector figures, previews, and LaTeX tables
│   ├── figure_01_reintermediation_trilemma.{pdf,png}
│   ├── figure_02_settlement_layer_comparison.{pdf,png}
│   ├── figure_03_ln_topology_schematic.{pdf,png}
│   ├── figure_04_ln_reliability_synthesis.{pdf,png}
│   ├── figure_05_security_budget_dilemma.{pdf,png}
│   ├── figure_06_mining_pool_concentration.{pdf,png}
│   ├── figure_A1_rolling_volatility.{pdf,png}
│   ├── figure_A2_var_comparison.{pdf,png}
│   ├── figure_A3_garch_volatility.{pdf,png}
│   ├── figure_A4_drawdowns.{pdf,png}
│   ├── figure_A5_correlation_dcc.{pdf,png}
│   ├── figure_B1_wash_trading_updated.{pdf,png}
│   ├── figure_B2_tether_dominance.{pdf,png}
│   ├── figure_C1_attack_cost_breakdown.{pdf,png}
│   ├── table_figure_manifest.{csv,tex}
│   ├── table_A_var_es_comparison.{csv,tex}
│   ├── table_A_var_backtests.{csv,tex}
│   ├── table_A_adf_stationarity.{csv,tex}
│   ├── table_garch_race_bitcoin.{csv,tex}
│   └── table_persistence_halflife_bitcoin.{csv,tex}
└── README.md                           # Documentation and replication instructions
```

---

## Figure & Table Manifest

| Label | File Basename | Evidentiary Status | Manuscript Role | Data Source / Method |
|:---|:---|:---|:---|:---|
| **Fig. 1** | `figure_01_reintermediation_trilemma` | `conceptual` | Framework (§2) | Synthesis diagram |
| **Fig. 2** | `figure_02_settlement_layer_comparison` | `reproduced` | Pillar I (§3) | Fed 2024 PFMI; ECB TARGET 2024; Visa/MC 10-K |
| **Fig. 3** | `figure_03_ln_topology_schematic` | `conceptual` | Pillar II (§4) | Graph topology theory (Avarikioti et al., 2020) |
| **Fig. 4** | `figure_04_ln_reliability_synthesis` | `reproduced` | Pillar II (§4) | Waugh & Holz (2020); River (2023); Lin (2020); Zabka (2022); Atmanaviciute (2025) |
| **Fig. 5** | `figure_05_security_budget_dilemma` | `theoretical-model` | Pillar III (§5) | Budish (2018) security budget scenarios |
| **Fig. 6** | `figure_06_mining_pool_concentration` | `reproduced` | Pillar III (§5) | Hashrate Index; B10C (2025); Spark (2026); SV2 audit |
| **Fig. A1**| `figure_A1_rolling_volatility` | `empirical` | Appendix A | Daily returns, native calendars (BTC: 365d, TradFi: 252d) |
| **Fig. A2**| `figure_A2_var_comparison` | `empirical` | Appendix A | Historical, Cornish-Fisher, and EVT (POT/GPD) VaR |
| **Fig. A3**| `figure_A3_garch_volatility` | `empirical` | Appendix A | BIC-selected conditional volatility specification |
| **Fig. A4**| `figure_A4_drawdowns` | `empirical` | Appendix A | Cumulative log max drawdowns |
| **Fig. A5**| `figure_A5_correlation_dcc` | `empirical` | Appendix A | DCC-GARCH(1,1) raw vs. DXY/VIX-residualized |
| **Fig. B1**| `figure_B1_wash_trading_updated` | `reproduced` | Appendix B | Bitwise (2019); Cong et al. (2022); Sila et al. (2025) |
| **Fig. B2**| `figure_B2_tether_dominance` | `reproduced` | Appendix B | CoinMarketCap; DeFiLlama aggregations |
| **Fig. C1**| `figure_C1_attack_cost_breakdown` | `reproduced` | Appendix C | Harvey (Oct 2025 static vs. Jul 2026 derivatives update) |
| **Tab. M** | `table_figure_manifest` | `audit` | Reproducibility | Machine-readable figure catalog |

---

## Replication Guide

### Environment Setup

Python 3.10+ is recommended. Install the required econometric and plotting libraries:

```bash
pip install yfinance arch statsmodels numpy pandas matplotlib scipy
```

### Execution

Run the interactive driver from the repository root:

```bash
python bitcoin_analysis_v7.py
```

The CLI menu options map directly to the paper's figures and appendices:

```
======================================================================
   v7 ANALYSIS -- RE-INTERMEDIATION TRILEMMA -- MAIN MENU
======================================================================
  [1]  Fig 1: The Re-Intermediation Trilemma (framework, NEW)
  [2]  Fig 2: Settlement-layer throughput (Pillar I)
  [3]  Fig 3: LN topology mesh vs. hub-and-spoke (Pillar II, theory)
  [4]  Fig 4: LN reliability synthesis (Pillar II, evidence)
  [5]  Fig 5: Security budget dilemma (Pillar III, theory)
  [6]  Fig 6: Mining pool concentration (Pillar III, evidence)
  [7]  Fig A1: Volatility comparison (Appendix A)
  [8]  Figs A2-A3: VaR/ES suite + GARCH race (Appendix A)
  [9]  Figs A4-A5: Drawdowns + DCC-GARCH correlation (Appendix A)
  [10] Table A: ADF stationarity report (Appendix A)
  [11] Fig B1: Wash-trading estimates (Appendix B)
  [12] Fig B2: Tether/stablecoin dominance (Appendix B)
  [13] Fig C1: 51% attack cost breakdown (Appendix C)
  [14] Figure manifest (evidentiary status + manuscript role)
  [15] Run all
  [0]  Exit
----------------------------------------------------------------------
```

* Select **`[15]`** to run all tests and export all publication figures (`.pdf`, `.png`) and tables (`.csv`, `.tex`) to the `paper_outputs/` directory.
* Run **`[14]`** to audit the manuscript figure-manifest classifications directly in the terminal.

---

## Citation

If referencing this analysis, empirical framework, or code in academic research, please cite:

```bibtex
@article{soleimani2026reintermediation,
  title   = {The Re-Intermediation Trilemma: Scaling, Liquidity, and Structural Centralization in Bitcoin},
  author  = {Soleimani, Hamoon},
  year    = {2026},
  month   = {September},
  note    = {Working paper and reproducible research archive}
}
```

## License
Released under the [MIT License](LICENSE). Models, data schemas, and figures may be reused with appropriate attribution.
