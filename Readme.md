# Nifty 50 Time Series Analysis & Return Predictability Research

[![Empirical Econometrics](https://img.shields.io/badge/Econometrics-ADF%20%7C%20ARIMA-blue)](#8-results--evaluation)
[![Stationarity](https://img.shields.io/badge/Returns%20Stationarity-p%20%3C%200.001-emerald)](#8-results--evaluation)
[![EMH Verification](https://img.shields.io/badge/EMH%20Baseline-Confirmed%20(Null%20Result)-purple)](#2-why-it-was-built)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

An empirical quantitative econometrics research notebook investigating the predictability of daily **Nifty 50 (`^NSEI`)** prices and returns (2014–2026), alongside long-term Indian macroeconomic inflation dynamics (1960–2026). The project rigorously tests whether autoregressive integrated moving average (ARIMA) models can forecast index return levels. The core finding is an empirical **null result** ($p > 0.05$), confirming the Efficient Market Hypothesis (EMH) at daily frequencies and establishing the foundational justification for volatility regime clustering in downstream platforms.

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. Generated Diagnostic Artifacts](#11-generated-diagnostic-artifacts)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Future Expansion](#15-roadmap--future-expansion)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

Given 12+ years of daily Nifty 50 market prices and 60+ years of Indian Consumer Price Index (CPI) historical data, this research notebook:

- **Tests Stationarity & Unit Roots**: Evaluates raw price levels and log-differenced returns using Augmented Dickey-Fuller (ADF) tests.
- **Deconstructs Time Series Components**: Splits the Nifty 50 series into secular trend, seasonal cycles, and residual noise via classical decomposition.
- **Analyzes Serial Correlation**: Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions up to 40 lags to test for memory structures.
- **Estimates ARIMA Models**: Fits parsimonious $ARIMA(1,0,1)$ specifications on stationary return series.
- **Evaluates Out-of-Sample Forecasts**: Quantifies forecast shrinkage and shows that linear predictions degenerate to the historical sample mean.
- **Contextualizes Macro Inflation**: Analyzes 60+ years of Indian CPI from `indianCPI_1960_feb26.csv` to contrast nominal equity gains against real purchasing power.

---

## 2. Why It Was Built

A common trap in quantitative finance is forcing machine learning algorithms onto raw price levels or daily returns expecting directional alpha. This research demonstrates systematically why that approach fails:

- **Prices are Non-Stationary**: Index levels exhibit stochastic drift and unit roots; training models directly on non-stationary price levels produces spurious regressions.
- **Returns Behave as Near-Random Walks**: At daily intervals, information is rapidly priced in. Past return signs provide virtually zero statistical leverage for predicting tomorrow's return direction.
- **Signal Lives in Variance, Not Mean Drift**: While daily directional mean return is unpredictable (white noise), return **variance** exhibits significant volatility clustering (Mandelbrot, 1963).
- **The Empirical Null Result as a Foundational Stepping Stone**: Proving the null result on first-moment return drift ($R^2 < 0.005$) provides the mathematical and economic justification for shifting focus to second-moment modeling (GARCH and Hidden Markov Models) in the Volatility Intelligence Platform.

---

## 3. System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION & SOURCING                       │
│                                                                        │
│   ├── Yahoo Finance API ──► Nifty 50 (^NSEI) Daily OHLCV (2014–2026)   │
│   └── Local CSV         ──► Indian CPI Series (1960–2026)              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     ECONOMETRIC ANALYSIS PIPELINE                      │
│                             (main.ipynb)                               │
│                                                                        │
│   1. Stationarity Testing (Augmented Dickey-Fuller)                    │
│      ├── Raw Prices:  ADF = -0.245 (p = 0.942)  ──► Non-Stationary     │
│      └── Log Returns: ADF = -56.82 (p < 0.001)  ──► Stationary I(1)    │
│                                                                        │
│   2. Classical Time Series Decomposition (Trend, Seasonality, Resid)   │
│                                                                        │
│   3. Serial Dependence Diagnostics (ACF & PACF up to 40 lags)          │
│      └── All lags fall within 95% Bartlett confidence bounds           │
│                                                                        │
│   4. ARIMA(1,0,1) Fitting & Forecast Evaluation                        │
│      └── Insignificant coefficients (p > 0.05), Forecast -> Sample Mean│
│                                                                        │
│   5. Macro CPI Inflation Contextualization                             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     GENERATED RESEARCH ARTIFACTS                       │
│                                                                        │
│   output/nifty_overview.png      output/decomposition.png              │
│   output/acf_pacf.png            output/arima_forecast.png             │
│   output/inflation_analysis.png                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.10` | Core programming runtime | High-performance ecosystem for scientific computing and econometrics |
| **statsmodels** | `^0.14.1` | Econometric modeling | Industry-standard suite for ADF unit-root tests, ACF/PACF, and ARIMA estimation |
| **yfinance** | `^0.2.36` | Market data ingestion | Direct, programmatic retrieval of 12+ years of Nifty 50 (`^NSEI`) adjusted prices |
| **pandas** | `^2.1.0` | Time series manipulation | Datetime indexing, log return differences, and CPI tabular parsing |
| **numpy** | `^1.26.0` | Numerical computing | Array manipulations, logarithmic transforms, and statistical aggregations |
| **matplotlib** | `^3.8.0` | Diagnostic plotting | Multi-panel diagnostic figures exported to publication PNG files |
| **seaborn** | `^0.13.0` | Statistical visualization | Density curves and distribution overlays for return series |
| **scikit-learn** | `^1.3.0` | Evaluation metrics | Baseline mean squared error and scoring routines |

---

## 5. Data

- **Financial Market Data**:
  - **Asset**: Nifty 50 Index (`^NSEI`).
  - **Source**: Yahoo Finance API.
  - **Horizon**: 2014 to 2026 (~3,000 daily trading sessions).
  - **Attributes**: Open, High, Low, Close, Adj Close, Volume.
- **Macroeconomic Inflation Data**:
  - **Asset**: Indian Consumer Price Index (CPI).
  - **Source**: `indianCPI_1960_feb26.csv`.
  - **Horizon**: 1960 to February 2026 (60+ years of monthly observations).
- **Preprocessing Applied**:
  - Continuous log return calculation: $r_t = \ln(P_t / P_{t-1})$.
  - First-differencing of non-stationary raw price levels.
  - DateTime alignment and missing value validation.

---

## 6. Step-by-Step Pipeline

1. **Market Data Retrieval**: Query Yahoo Finance for `^NSEI` daily data from 2014 to present; verify data integrity and absence of missing trading days.
2. **Stationarity & Unit Root Testing**: Run Augmented Dickey-Fuller tests on raw prices and continuous log returns to verify integration order $I(1)$.
3. **Time Series Decomposition**: Apply additive decomposition separating price series into secular multi-year trend, seasonal patterns, and residual noise.
4. **Autocorrelation Analysis**: Compute and plot ACF and PACF across 40 daily lags with 95% Bartlett confidence bounds ($\pm 1.96/\sqrt{N}$).
5. **ARIMA Model Specification**: Fit parsimonious $ARIMA(1,0,1)$ model:
   $$r_t = c + \phi_1 r_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
6. **Coefficient Testing & Forecast Evaluation**: Evaluate statistical significance of $\phi_1$ and $\theta_1$; generate out-of-sample forecast and demonstrate shrinkage to historical mean.
7. **Macroeconomic CPI Analysis**: Parse `indianCPI_1960_feb26.csv` and visualize secular inflation curves across six decades to provide macroeconomic context.
8. **Artifact Generation**: Export all diagnostic plots to the `output/` directory.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **Raw price series is non-stationary** (unit root present) | Cannot apply ARIMA or linear time-series econometric models directly on raw price levels — producing spurious regressions and invalid statistical inferences | Applied **first-differencing** via continuous log returns ($r_t = \ln(P_t/P_{t-1})$), which the Augmented Dickey-Fuller test confirmed is stationary at the 1% significance level ($p < 0.001$, ADF statistic $-56.82$) |
| **ARIMA coefficients were statistically insignificant** | Both AR(1) and MA(1) parameters had $p > 0.05$, meaning the model could not detect linear predictive structure in daily returns | Accepted the **null result as the core empirical finding** — this validates the Efficient Market Hypothesis (EMH). Daily returns behave as near-random walks, motivating the architectural pivot toward second-moment volatility modeling |
| **Overfitting temptation** with higher-order ARIMA(p,d,q) | Fitting ARIMA(5,1,5) or higher orders would capture transient noise and create false illusion of in-sample explanatory power | Deliberately restricted to **parsimonious ARIMA(1,0,1)** — the minimal non-trivial specification. ACF and PACF diagnostics confirmed zero statistically significant serial correlation beyond lag 0, proving higher-order terms overfit microstructure noise |
| **ARIMA forecast collapsed to the sample mean** | Out-of-sample directional predictions flatlined at near-zero drift while realized market prices fluctuated wildly | Documented this expected econometric reality as visual proof (saved as `arima_forecast.png`), confirming that return-level forecasting is futile and directing quantitative attention to variance-process modeling (GARCH, HMM) |
| **CPI data had inconsistent formatting** (1960–2026) | 60+ years of Indian CPI data had varying date formats, frequency adjustments, and column naming conventions | Cleaned and standardized the local dataset into `indianCPI_1960_feb26.csv`, maintaining descriptive macroeconomic analysis rather than forcing fragile cointegration models on mixed frequencies |

---

## 8. Results & Evaluation

### Stationarity & Unit Root Test Results (ADF)

| Series | ADF Statistic | p-value | 1% Critical Value | Econometric Conclusion |
|---|:---:|:---:|:---:|---|
| **Raw Nifty 50 Closing Price** | $-0.245$ | **0.942** | $-3.435$ | **Non-Stationary (Unit Root Present, $I(1)$)** |
| **Log Returns ($r_t$)** | **$-56.821$** | **0.000** | $-3.435$ | **Stationary ($I(0)$)** |

### ARIMA(1,0,1) Estimation Summary

- **$\phi_1$ (Autoregressive Lag 1)**: Statistically insignificant ($p > 0.05$).
- **$\theta_1$ (Moving Average Lag 1)**: Statistically insignificant ($p > 0.05$).
- **Model In-Sample $R^2$**: $< 0.005$ (explains less than 0.5% of variance).
- **Out-of-Sample Forecast**: Degenerates to historical sample mean with wide unbounded forecast confidence bands.

### Econometric Interpretation

1. **Weak-Form Efficient Market Hypothesis Confirmed**: Past price information contains no exploitable linear signal for forecasting future daily return drift on the Nifty 50 index.
2. **Justification for Non-Linear / Regime-Based Modeling**: Because the first moment ($\mathbb{E}[r_t]$) is white noise, predictive modeling must shift to the second moment ($\text{Var}(r_t)$), where strong volatility clustering and regime persistence exist.

---

## 9. Project Structure

```text
Nifty-Time-Series/
├── main.ipynb                  # Primary research notebook (EDA, ADF, ARIMA, CPI)
├── indianCPI_1960_feb26.csv    # Indian Consumer Price Index dataset (1960–2026)
├── requirements.txt            # Python dependencies (statsmodels, yfinance, etc.)
├── Readme.md                   # Project documentation
└── output/                     # Generated diagnostic visualizations
    ├── nifty_overview.png      # 12-year price trajectory & daily returns
    ├── decomposition.png       # Seasonal and trend decomposition
    ├── acf_pacf.png            # Autocorrelation and Partial Autocorrelation plots
    ├── arima_forecast.png      # Out-of-sample ARIMA forecast vs actual realization
    └── inflation_analysis.png  # CPI historical series and inflation trends
```

---

## 10. Getting Started

### Prerequisites

- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

### 1. Environment Setup

```bash
git clone https://github.com/RaajitSingh1306/Nifty-Time-Series.git
cd "Nifty-Time-Series"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Notebook

```bash
jupyter notebook main.ipynb
# or
jupyter lab
```

Select **Run All Cells**. The notebook will:
1. Fetch latest daily OHLCV market data for Nifty 50 (`^NSEI`) via Yahoo Finance.
2. Load local CPI inflation records from `indianCPI_1960_feb26.csv`.
3. Compute ADF statistics, decomposition, ACF/PACF graphs, and ARIMA models.
4. Export updated diagnostic figures to the `output/` directory.

---

## 11. Generated Diagnostic Artifacts

| Artifact | Description |
|---|---|
| `output/nifty_overview.png` | Dual-panel visualization showing 12-year Nifty 50 closing price curve and daily log return oscillations. |
| `output/decomposition.png` | Additive decomposition separating price history into secular trend, seasonal fluctuations, and residual variance. |
| `output/acf_pacf.png` | Bartlett correlation bounds showing returns are indistinguishable from white noise up to lag 40. |
| `output/arima_forecast.png` | Visual proof of forecast shrinkage: ARIMA projections flatline at zero drift while realized prices fluctuate. |
| `output/inflation_analysis.png` | 60+ year timeline of Indian inflation, demonstrating the real versus nominal purchasing power expansion of equity markets. |

---

## 12. Deployment

As an econometric research notebook, this project does not run as a persistent server. It serves as an offline research artifact and analytical benchmark for production quant systems.

---

## 13. Connected Portfolio Projects

```text
Nifty-Time-Series (Proves return levels are unpredictable)
       │
       ▼
Volatility Intelligence Platform (Models return variance & volatility clustering)
       ├── GARCH(1,1) conditional volatility
       ├── 3-state Gaussian Hidden Markov Model (unsupervised regime discovery)
       ├── XGBoost forward-looking predictive layer (AUC 0.7453)
       └── Next.js 14 Web Dashboard + FastAPI service
       │
       ▼
Nifty Sector Rotation Strategy (Exploits cross-sectional relative momentum)
```

- **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: Flagship ML platform directly motivated by this null result; models second-moment volatility clustering using GARCH and HMM.
- **[Nifty Sector Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation)**: Shifts from single-index time-series prediction to cross-sectional relative momentum across 10 sectors.
- **[Finance KPI](https://github.com/RaajitSingh1306/Finance_Kpi)**: Single-ticker KPI engine sharing the same continuous log return formulation.

---

## 14. Limitations & Known Issues

- **Linear Model Restriction**: Evaluates linear autoregression (ARIMA) only; does not test non-linear recurrent networks (LSTM/GRU) or Transformer architectures in this notebook.
- **No Dual-Stationarity Confirmation**: Relies exclusively on ADF without running the complementary KPSS test (where the null hypothesis is stationarity).
- **Descriptive Inflation Analysis**: The Indian CPI exploration is descriptive; it does not perform Johansen cointegration tests between inflation indices and equity multiples.
- **Single Index Focus**: Analysis is restricted to the Nifty 50 benchmark (`^NSEI`); does not contrast findings against small-cap indices or commodities.

---

## 15. Roadmap / Future Expansion

- [ ] **Dual Stationarity Testing**: Incorporate KPSS test suite side-by-side with ADF to classify series into trend-stationary, difference-stationary, or ambiguous.
- [ ] **Johansen Cointegration & VECM**: Implement Vector Error Correction Models (VECM) between Indian CPI, G-Sec yields, and Nifty 50.
- [ ] **GARCH Residual Bridge**: Add GARCH(1,1) diagnostic code directly to demonstrate the shift from return-level randomness to second-moment volatility clustering.
- [ ] **Deep Learning Baseline**: Implement an LSTM baseline to empirically prove that deep neural networks also fail to beat the random-walk benchmark on daily raw returns.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This research is conducted solely for quantitative econometrics education and portfolio analysis. It does not constitute investment advice or trading recommendations.
