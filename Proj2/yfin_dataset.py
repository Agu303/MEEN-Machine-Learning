import yfinance as yf

# BTC-USD: Bitcoin price in USD from Yahoo Finance
btc = yf.download("BTC-USD", start="2017-01-01", end="2025-04-01")
btc.to_csv("btc_usd_yfinance.csv")
