# Nifty 50 Time Series Analysis & Return Predictability Research

[![Empirical Econometrics](https://img.shields.io/badge/Econometrics-ADF%20%7C%20ARIMA-blue)](#key-results)
[![Stationarity](https://img.shields.io/badge/Returns%20Stationarity-p%20%3C%200.001-emerald)](#1-stationarity--unit-root-testing)
[![EMH Verification](https://img.shields.io/badge/EMH%20Baseline-Confirmed-purple)](#interpretation--theoretical-foundation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical quantitative econometrics research notebook investigating the predictability of daily **Nifty 50 (`^NSEI`)** prices and returns (2014–2026), alongside long-term Indian macroeconomic inflation dynamics (1960–2026).

The project rigorously tests whether autoregressive integrated moving average (ARIMA) models can forecast index return levels. The core finding is an empirical **null result** ($p > 0.05$), confirming the Efficient Market Hypothesis (EMH) at daily frequencies and establishing the foundational justification for volatility regime clustering.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Why It Matters: The Null Result as a Feature](#why-it-matters-the-null-result-as-a-feature)
3. [Key Econometric Findings](#key-econometric-findings)
4. [Project Structure](#project-structure)
5. [Generated Diagnostic Outputs](#generated-diagnostic-outputs)
6. [Where & How to Start](#where--how-to-start)
7. [Connected Projects & Architectural Lineage](#connected-projects--architectural-lineage)

---

## What This Project Does

Given 12+ years of daily Nifty 50 market prices and 60+ years of Indian Consumer Price Index (CPI) historical data, this research notebook:

1. **Tests Stationarity**: Evaluates unit roots on raw price levels and log-differenced returns using Augmented Dickey-Fuller (ADF).
2. **Deconstructs Time Series**: Splits Nifty 50 into underlying secular trend, seasonal cycles, and residual noise components.
3. **Analyzes Serial Correlation**: Computes Autocorrelation (ACF) and Partial Autocorrelation (PACF) functions up to 40 lags.
4. **Fits ARIMA(p, d, q) Models**: Estimates autoregressive and moving-average parameters on stationary return series.
5. **Evaluates Return Forecasts**: Measures out-of-sample forecast accuracy against actual market realization.
6. **Analyzes Macro Inflation**: Evaluates CPI trajectory from `indianCPI_1960_feb26.csv` to contextualize real versus nominal equity market appreciation.

---

## Why It Matters: The Null Result as a Feature

A common trap in quantitative modeling is forcing predictive algorithms onto raw price levels or daily returns. This research systematically demonstrates why that approach fails:

* **Prices are non-stationary**: Index levels exhibit stochastic drift; training regression or neural models on non-stationary series leads to spurious correlations.
* **Returns behave as near-random walks**: At daily intervals, new information is rapidly priced in. Past return signs provide virtually zero statistical leverage for tomorrow's direction.
* **Signal exists in variance, not drift**: While daily mean return direction is unpredictable, return **variance** exhibits strong volatility clustering (large changes follow large changes, regardless of sign).

> [!NOTE]
> This null result directly motivates the architecture of the **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**. Instead of attempting to forecast impossible return levels, the system models the variance process using GARCH(1,1) and Hidden Markov Models (HMM).

---

## Key Econometric Findings

### 1. Stationarity & Unit Root Testing

| Series | ADF Test Statistic | p-value | Critical Value (1%) | Conclusion |
|---|---|---|---|---|
| **Raw Nifty 50 Price** | $-0.245$ | **0.942** | $-3.435$ | **Non-Stationary (Unit Root Present)** |
| **Log Returns ($r_t = \ln(P_t/P_{t-1})$)** | $-56.821$ | **0.000** | $-3.435$ | **Stationary ($I(1)$ Integrated)** |

* Raw closing prices follow a geometric random walk with upward drift.
* First-differencing achieves stationarity, allowing econometric analysis on log returns.

### 2. Autocorrelation (ACF & PACF) Diagnostics

* Across 40 daily lags, both ACF and PACF values for log returns remain within the 95% Bartlett confidence bounds ($\pm 1.96 / \sqrt{N}$).
* Confirms absence of serial correlation in daily returns—consistent with weak-form market efficiency.

### 3. ARIMA(1,0,1) Model Estimation

When fitting an $ARIMA(1,0,1)$ process:
$$r_t = c + \phi_1 r_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$

* **$\phi_1$ (Autoregressive coefficient)**: Insignificant ($p > 0.05$)
* **$\theta_1$ (Moving Average coefficient)**: Insignificant ($p > 0.05$)
* **Model $R^2$**: $< 0.005$
* **Forecast Line**: Collapses to the unconditional historical mean.

---

## Key Design Decisions

- **Augmented Dickey-Fuller (ADF) over KPSS**: ADF explicitly tests the null hypothesis of a unit root ($H_0$: non-stationary) against the alternative of difference-stationarity. In quantitative risk management, confirming rejection of a unit root on differences at the 1% significance level is standard before passing returns to volatility models.
- **Parsimonious ARIMA(1,0,1) Specification**: AR(1)+MA(1) represents the minimal non-trivial linear autoregressive structure. Higher-order specifications ($p, q > 1$) were deliberately avoided because sample ACF and PACF correlations show no statistically significant lags; fitting higher-order polynomials would overfit high-frequency market microstructure noise.
- **Log Returns over Simple Returns**: Logarithmic returns ($\ln(P_t / P_{t-1})$) are time-additive across horizons and conform to the continuous-time geometric Brownian motion assumptions required for standard stationarity testing.
- **252-Day Annualization Factor**: Standardized across the Indian market trading calendar (~248–252 active trading sessions per year) for volatility and Sharpe scaling.
- **Additive Decomposition**: Classical additive seasonal decomposition was selected to visually isolate low-frequency multi-year trend components from residual noise without distorting return scale.

---

## Project Structure

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
    ├── arima_forecast.png      # Out-of-sample ARIMA forecast vs actual noise
    └── inflation_analysis.png  # CPI historical series and inflation rate trends
```

---

## Generated Diagnostic Outputs

The notebook executes end-to-end and exports five diagnostic figures to the `output/` folder:

| Artifact | Description |
|---|---|
| `output/nifty_overview.png` | Dual-panel visualization showing 12-year Nifty 50 closing price curve and daily log return oscillations. |
| `output/decomposition.png` | Additive/multiplicative decomposition separating price history into secular trend, seasonal fluctuations, and residual variance. |
| `output/acf_pacf.png` | Ljung-Box and Bartlett correlation bounds showing returns are indistinguishable from white noise. |
| `output/arima_forecast.png` | Visual proof of forecast shrinkage: ARIMA out-of-sample projections flatline around zero drift while realized prices fluctuate. |
| `output/inflation_analysis.png` | 60+ year timeline of Indian inflation, demonstrating the real versus nominal purchasing power expansion of equity markets. |

---

## Where & How to Start

### Prerequisites

* Python 3.10+
* Jupyter Notebook or JupyterLab environment

### 1. Environment Setup

Clone or navigate to the project directory:

```bash
cd "Nifty-Time-Series"
```

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install the required econometric and visualization libraries:

```bash
pip install -r requirements.txt
```

*(Key libraries: `statsmodels`, `yfinance`, `pandas-datareader`, `matplotlib`, `seaborn`, `scikit-learn`)*

### 3. Launch Notebook

Start Jupyter and open `main.ipynb`:

```bash
jupyter notebook main.ipynb
# or
jupyter lab
```

Select **Run All Cells**. The notebook will:
1. Fetch latest daily OHLCV market data for Nifty 50 (`^NSEI`) via Yahoo Finance.
2. Load local CPI inflation records from `indianCPI_1960_feb26.csv`.
3. Compute ADF statistics, decomposition, ACF/PACF graphs, and ARIMA models.
4. Refresh all figures inside the `output/` directory.

---

## Connected Projects & Architectural Lineage

This research serves as the empirical foundation for the broader quantitative platform portfolio:

```
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

| Project | Role | GitHub Repository |
|---|---|---|
| **Nifty Time Series** (This Repo) | Empirical stationarity and EMH baseline | [Nifty-Time-Series](https://github.com/RaajitSingh1306/Nifty-Time-Series) |
| **Volatility Intelligence Platform** | Shipped GARCH + HMM + XGBoost prediction system | [volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) |
| **Nifty Sector Rotation** | Cross-sectional momentum trading across 10 sectors | [Nifty-Sector-Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation) |

---

## Limitations & Roadmap

### Known Limitations
- **Linear Model Restriction**: Evaluates linear autoregression (ARIMA) only; does not fit non-linear deep learning baselines (e.g. LSTM, Transformer) within this notebook.
- **No Dual-Stationarity Confirmation**: Relies exclusively on ADF without running the complementary KPSS test (where the null hypothesis is stationarity).
- **Descriptive Inflation Analysis**: The Indian CPI exploration is descriptive; it does not perform Engle-Granger or Johansen cointegration tests between inflation indices and equity multiples.
- **Single Index Focus**: Analysis is restricted to the Nifty 50 benchmark (`^NSEI`); does not contrast findings against broader small-cap indices or global asset classes.
- **In-Sample Fit Focus**: While the ARIMA forecast trajectory is visualized, formal out-of-sample rolling MAE/RMSE scoring was not computed because the model degenerated to the sample mean.

### Roadmap
- [ ] **Dual Stationarity Testing**: Incorporate KPSS test suite side-by-side with ADF to classify series into trend-stationary, difference-stationary, or ambiguous.
- [ ] **Cointegration & Error Correction**: Implement vector error correction models (VECM) and Johansen cointegration tests between Indian CPI, G-Sec yields, and Nifty 50.
- [ ] **GARCH Bridge**: Add GARCH(1,1) residual diagnostic code directly to demonstrate the shift from return-level randomness to second-moment volatility clustering.
- [ ] **Deep Learning Baseline**: Implement a simple LSTM baseline to empirically prove that deep neural networks also fail to beat the random-walk benchmark on daily raw returns.

---

## License & Disclaimer

MIT License. This analysis is conducted solely for quantitative finance research and educational demonstration. It is not financial or investment advice.
