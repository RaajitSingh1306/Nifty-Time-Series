# Nifty 50 — Time Series Forecast & Macro Analysis

## Overview
Time series analysis and forecasting on Nifty 50 index prices 
combined with Indian inflation data as a macro indicator.

## Dataset
- Nifty 50 daily OHLCV data, 10 April 2014 – 10 March 2026 (2927 rows)
- Source: Yahoo Finance via yfinance
- Macro indicator: India CPI inflation (World Bank, annual)

## Key Findings
- Raw prices are non-stationary (ADF p=0.94); daily returns are 
  stationary (ADF p=0.00)
- ARIMA(1,0,1) parameters were statistically insignificant — 
  confirms Efficient Market Hypothesis for Nifty returns
- Inflation weakly correlates with Nifty returns (r=0.39, n=12) 
  but too unstable to be a reliable standalone predictor
- Model honestly captured long-term drift but cannot predict 
  short-term volatility

## What I Learned
- Reading the price and return chart 
- Understood how bad of an indicator inflation is for the entire market
- Long term trading is better than day trade due to lower fluctuation and eventual rise in market

## Files
- nifty_forecast.ipynb — full analysis
- outputs/nifty_overview.png — price and returns chart
- outputs/decomposition.png — trend/seasonality/residual
- outputs/acf_pacf.png — autocorrelation analysis
- outputs/arima_forecast.png — forecast vs actual
- outputs/inflation_analysis.png — macro indicator analysis

## Setup
pip install yfinance pandas-datareader numpy matplotlib seaborn statsmodels numpy scikit-learn
```