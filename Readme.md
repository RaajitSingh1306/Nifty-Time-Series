# Nifty 50 Time Series Analysis & Return Predictability Research

[![Empirical Econometrics](https://img.shields.io/badge/Econometrics-ADF%20%7C%20ARIMA-blue)](#8-results--evaluation)
[![Stationarity](https://img.shields.io/badge/Returns%20Stationarity-p%20%3C%200.001-emerald)](#8-results--evaluation)
[![EMH Verification](https://img.shields.io/badge/EMH%20Baseline-Consistent%20(Null%20Result)-purple)](#2-why-it-was-built)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

An empirical quantitative econometrics research notebook investigating the predictability of daily **Nifty 50 (`^NSEI`)** prices and returns (2014–2026), alongside annual Indian CPI inflation (analysed for 2014–2026). The project rigorously tests whether autoregressive integrated moving average (ARIMA) models can forecast index return levels. The core finding is an empirical **null result** ($p > 0.05$), consistent with the weak-form Efficient Market Hypothesis (EMH) at daily frequencies and establishing the foundational justification for volatility regime clustering in downstream platforms.

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

Given 12 years of daily Nifty 50 closes (2014 to 2026) and annual Indian CPI inflation (World Bank, 1960 to 2026; only 2014 to 2026 is analysed), this research notebook:

- **Tests Stationarity & Unit Roots**: Evaluates raw price levels and simple daily returns using Augmented Dickey-Fuller (ADF) tests.
- **Deconstructs Time Series Components**: Splits the Nifty 50 series into secular trend, seasonal and residual components of weekly-average closes via classical multiplicative decomposition (period = 52); the seasonal component is weak and likely an artefact.
- **Analyzes Serial Correlation**: Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions up to 30 lags to test for memory structures.
- **Estimates ARIMA Models**: Fits parsimonious $ARIMA(1,0,1)$ specifications on stationary return series.
- **Evaluates Out-of-Sample Forecasts**: Quantifies forecast shrinkage and shows that linear predictions degenerate to the historical sample mean.
- **Contextualizes Macro Inflation**: Compares annual Indian CPI inflation (2014 to 2026, from `indianCPI_1960_feb26.csv`) with yearly average Nifty returns (13 points: r = +0.381; r = +0.190 for 2014 to 2025, which drops only the partial year 2026; r = -0.040 for 2015 to 2025; none significant at 5%, so sign and size depend on which years are included; descriptive only).

---

## 2. Why It Was Built

A common trap in quantitative finance is forcing machine learning algorithms onto raw price levels or daily returns expecting directional alpha. This research demonstrates systematically why that approach fails:

- **Prices are Non-Stationary**: Index levels exhibit stochastic drift and unit roots; training models directly on non-stationary price levels produces spurious regressions.
- **Returns Behave as Near-Random Walks**: At daily intervals, information is rapidly priced in. Past return signs provide virtually zero statistical leverage for predicting tomorrow's return direction.
- **Signal Lives in Variance, Not Mean Drift**: While daily directional mean return is unpredictable (out-of-sample R-squared about 0), return **variance** exhibits significant volatility clustering (Mandelbrot, 1963).
- **The Empirical Null Result as a Foundational Stepping Stone**: Proving the null result on first-moment return drift (R-squared about 0) provides the mathematical and economic justification for shifting focus to second-moment modeling (GARCH and Hidden Markov Models) in the Volatility Intelligence Platform.

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
│      ├── Raw Prices:  ADF = +0.059 (p = 0.963)  ──► Non-Stationary     │
│      └── Returns:     ADF = -15.36 (p < 0.001)  ──► Stationary I(0)    │
│                                                                        │
│   2. Classical Time Series Decomposition (Trend, Seasonality, Resid)   │
│                                                                        │
│   3. Serial Dependence Diagnostics (ACF & PACF up to 30 lags)          │
│      └── 8/30 ACF, 7/30 PACF lags outside 95% bounds                   │
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

| Library / Tool | Tested version | Purpose | Rationale |
| --- | --- | --- | --- |
| **Python** | `3.15.0rc2` (release candidate; every version in this table is recorded in `output/results.json` under env) | Core programming runtime | High-performance ecosystem for scientific computing and econometrics |
| **statsmodels** | `0.15.0` | Econometric modeling | Industry-standard suite for ADF unit-root tests, ACF/PACF, and ARIMA estimation |
| **yfinance** | `1.8.0` | Market data ingestion | Direct, programmatic retrieval of 12 years of Nifty 50 (`^NSEI`) daily closes |
| **pandas** | `3.1.1` | Time series manipulation | Datetime indexing, daily return calculation, and CPI tabular parsing |
| **numpy** | `2.5.5` | Numerical computing | Array manipulations, logarithmic transforms, and statistical aggregations |
| **matplotlib** | `3.11.2` | Diagnostic plotting | Multi-panel diagnostic figures exported to publication PNG files |
| **scikit-learn** | `1.9.1` | Evaluation metrics | Baseline mean squared error and scoring routines |

---

## 5. Data

- **Financial Market Data**:
  - **Asset**: Nifty 50 Index (`^NSEI`).
  - **Source**: Yahoo Finance API.
  - **Horizon**: 2014-01-02 to 2026-02-27 (2,988 daily closes, 2,987 returns; fixed window).
  - **Attributes**: Open, High, Low, Close, Volume (yfinance auto_adjust=True; only Close is used).
- **Macroeconomic Inflation Data**:
  - **Asset**: Indian Consumer Price Index (CPI).
  - **Source**: `indianCPI_1960_feb26.csv`.
  - **Horizon**: 1960 to 2026, annual observations (World Bank indicator FP.CPI.TOTL.ZG); only 2014 to 2026 is used, with a caveat: the 2025 (2.8) and 2026 (3.2) values look like estimates, not published actuals (every other year has 9 decimals, and the file is dated Feb 2026, so a full-year 2026 figure cannot have been observed); their source was not verified.
- **Preprocessing Applied**:
  - Simple daily return: $r_t = P_t / P_{t-1} - 1$ (pandas pct_change).
  - Conversion of non-stationary raw price levels to simple daily returns.
  - DateTime alignment and missing value validation.

---

## 6. Step-by-Step Pipeline

1. **Market Data Retrieval**: Query Yahoo Finance for `^NSEI` daily data for the fixed window 2014-01-01 to 2026-03-01 (last close 2026-02-27); check for missing values.
2. **Stationarity & Unit Root Testing**: Run Augmented Dickey-Fuller tests on raw prices and simple daily returns to verify integration order $I(1)$.
3. **Time Series Decomposition**: Apply multiplicative decomposition (statsmodels seasonal_decompose, period = 52) to weekly-average closes, separating trend, seasonal and residual components.
4. **Autocorrelation Analysis**: Compute and plot ACF and PACF across 30 daily lags with 95% white-noise bounds ($\pm 1.96/\sqrt{N}$). Bar counts are computed against this flat bound; the shaded ACF band in the plot is the statsmodels default Bartlett band, which widens with lag, so counts read off the plot can differ.
5. **ARIMA Model Specification**: Fit parsimonious $ARIMA(1,0,1)$ model:
   $$r_t = c + \phi_1 r_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
6. **Coefficient Testing & Forecast Evaluation**: Evaluate statistical significance of $\phi_1$ and $\theta_1$; generate out-of-sample forecast and demonstrate shrinkage to historical mean.
7. **Macroeconomic CPI Analysis**: Parse `indianCPI_1960_feb26.csv` and compare 2014 to 2026 annual inflation with yearly average Nifty returns (descriptive; only 2026 is a partial year: 2014 has 240 daily closes, 2026 has 40).
8. **Artifact Generation**: Export all diagnostic plots to the `output/` directory.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
| --- | --- | --- |
| **Raw price series is non-stationary** (unit root present) | Cannot apply ARIMA or linear time-series econometric models directly on raw price levels — producing spurious regressions and invalid statistical inferences | Converted prices to **simple daily returns** ($r_t = P_t/P_{t-1} - 1$), which the Augmented Dickey-Fuller test confirmed is stationary at the 1% significance level ($p < 0.001$, ADF statistic $-15.36$, 11 lags) |
| **ARIMA coefficients were statistically insignificant** | Both AR(1) and MA(1) parameters had $p > 0.05$, meaning the model could not detect linear predictive structure in daily returns | Accepted the **null result as the core empirical finding** — this is consistent with the weak-form Efficient Market Hypothesis (EMH). Daily returns behave as near-random walks, motivating the architectural pivot toward second-moment volatility modeling |
| **Overfitting temptation** with higher-order ARIMA(p,d,q) | Fitting ARIMA(5,1,5) or higher orders would capture transient noise and create false illusion of in-sample explanatory power | Deliberately restricted to **parsimonious ARIMA(1,0,1)** — the minimal non-trivial specification. ACF/PACF show small but detectable serial correlation (8 of 30 ACF and 7 of 30 PACF bars outside the 95% bounds; lag-1 ACF -0.014), far too small to give a usable forecast, so higher orders were not pursued |
| **ARIMA forecast collapsed to the sample mean** | Out-of-sample directional predictions flatlined at near-zero drift while realized market prices fluctuated wildly | Documented this expected econometric reality as visual proof (saved as `arima_forecast.png`), confirming that return-level forecasting is futile and directing quantitative attention to variance-process modeling (GARCH, HMM) |

---

## 8. Results & Evaluation

### Stationarity & Unit Root Test Results (ADF)

| Series | ADF Statistic | p-value | 1% Critical Value | Econometric Conclusion |
| --- | :---: | :---: | :---: | --- |
| **Raw Nifty 50 Closing Price** | $+0.059$ | **0.963** | $-3.433$ | **Non-Stationary (Unit Root Present, $I(1)$)** |
| **Daily Returns ($r_t$)** | **$-15.362$** | **0.000** | $-3.433$ | **Stationary ($I(0)$)** |

ADF run (window 2014-01-02 to 2026-02-27; 2,988 prices, 2,987 returns; statsmodels default autolag=AIC): prices 0 lags, statistic +0.059; returns 11 lags, statistic -15.36 (-55.40 if forced to 0 lags). The statistic depends on lag choice; the rejection (p near 0) does not; the 1% critical value is -3.4325 for both series (saved as adf_crit_1pct). Source: output/results.json.

### ARIMA(1,0,1) Estimation Summary

- **$\phi_1$ (Autoregressive Lag 1)**: Statistically insignificant ($p = 0.710$).
- **$\theta_1$ (Moving Average Lag 1)**: Statistically insignificant ($p = 0.704$).
- **Model R-squared**: in-sample -2.5e-5; out-of-sample vs train mean 3.8e-6 (both about 0; source: output/results.json).
- **Out-of-Sample Forecast**: Degenerates to the train-sample mean (out-of-sample R-squared vs train mean about 0).
- **Out-of-Sample RMSE** (chronological 80/20 split): ARIMA 0.007992, train-mean forecast 0.007992, zero forecast 0.008004; ARIMA is 0.15% better than a zero forecast. Source: `output/results.json`.

### Econometric Interpretation

1. **Consistent with Weak-Form EMH**: Past returns carry no usable linear signal (out-of-sample R-squared about 0), though lag-1 autocorrelation is statistically detectable (-0.014; Ljung-Box lag 10 p = 1.5e-8).
2. **Justification for Non-Linear / Regime-Based Modeling**: Because the first moment ($\mathbb{E}[r_t]$) is close to unpredictable (out-of-sample R-squared about 0), predictive modeling must shift to the second moment ($\text{Var}(r_t)$), where volatility clustering is detectable (Ljung-Box on squared returns, lag 10: p about 0, below float precision).

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
    └── inflation_analysis.png  # Yearly average Nifty return vs CPI inflation, 2014 to 2026
```

---

## 10. Getting Started

### Prerequisites

- Python: the recorded run used 3.15.0rc2 (a release candidate); exact package versions are in requirements.txt
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

1. Fetch Nifty 50 (`^NSEI`) daily data for the fixed window 2014-01-01 to 2026-03-01 via Yahoo Finance.
2. Load local CPI inflation records from `indianCPI_1960_feb26.csv`.
3. Compute ADF statistics, decomposition, ACF/PACF graphs, and ARIMA models.
4. Export diagnostic figures and `output/results.json` (source of the numbers quoted here) to `output/`. Use Restart and Run All so cells execute in order (rerun = reopen main.ipynb and Restart and Run All; results.json is rewritten and its env block records the Python and package versions used). The data window is fixed at 2014-01-01 to 2026-03-01.

---

## 11. Generated Diagnostic Artifacts

| Artifact | Description |
| --- | --- |
| `output/nifty_overview.png` | Dual-panel visualization showing 12-year Nifty 50 closing price curve and daily simple return oscillations. |
| `output/decomposition.png` | Multiplicative decomposition of weekly-average closes (period 52) into trend, seasonal and residual components; seasonality is weak and likely an artefact. |
| `output/acf_pacf.png` | Bounds over 30 lags: 8 of 30 ACF and 7 of 30 PACF bars fall outside; lag-1 ACF is -0.014 (economically negligible). |
| `output/arima_forecast.png` | Visual proof of forecast shrinkage: ARIMA projections flatline at zero drift while realized prices fluctuate. |
| `output/inflation_analysis.png` | Yearly average Nifty return vs Indian CPI inflation, 2014 to 2026 (13 points; r = +0.381, falling to -0.040 on 2015 to 2025; descriptive). |

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
- **[Finance KPI](https://github.com/RaajitSingh1306/Finance_Kpi)**: Single-ticker KPI engine using the same simple-return formulation.

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
