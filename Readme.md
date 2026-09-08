# Nifty 50 ARIMA Forecast — Research Note

An exploratory analysis of Nifty 50 daily prices testing whether ARIMA can forecast 
index returns. The null result — it cannot — is the finding.

## Key Results

- **ADF test on raw prices**: p = 0.94 → non-stationary (prices have a unit root)
- **ADF test on log-returns**: p = 0.00 → stationary (returns are I(1))
- **ARIMA(1,0,1) on returns**: both AR and MA coefficients statistically insignificant
  (p > 0.05) — consistent with the Efficient Market Hypothesis at the daily frequency

## Interpretation

The result is not a modelling failure — it is the expected finding from decades of 
empirical finance research. Nifty 50 daily returns behave as a near-random walk at 
this horizon. Forecasting *price levels* is the wrong target; forecasting *volatility 
regimes* is where signal exists.

This analysis motivates the approach in 
[volatility-classifier-simplified](https://github.com/RaajitSingh1306/volatility-classifier-simplified), 
which models volatility clusters (GARCH + HMM) rather than return levels.

## Stack
`statsmodels` · `yfinance` · `pandas-datareader` · `matplotlib` · `seaborn`
