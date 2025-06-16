import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp

def fetch_price_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    if isinstance(tickers, str) or len(tickers) == 1:
        ticker = tickers[0] if isinstance(tickers, list) else tickers
        try:
            return data['Adj Close'].to_frame(name=ticker)
        except KeyError:
            return data['Close'].to_frame(name=ticker)

    adj_close_data = {}
    for ticker in tickers:
        try:
            adj_close_data[ticker] = data[ticker]['Adj Close']
        except KeyError:
            try:
                adj_close_data[ticker] = data[ticker]['Close']
            except KeyError:
                print(f"[WARN] No data for ticker: {ticker}")
                continue

    if not adj_close_data:
        raise ValueError("No valid price data found for any ticker.")

    return pd.DataFrame(adj_close_data).dropna()


def mean_variance_optimization(prices):
    returns = prices.pct_change().dropna()
    mu = returns.mean()
    cov = returns.cov()

    n = len(mu)
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.values @ w
    risk = cp.quad_form(w, cov.values)

    prob = cp.Problem(cp.Maximize(ret - gamma * risk),
                      [cp.sum(w) == 1, w >= 0])
    gamma.value = 0.1
    prob.solve()

    return w.value, mu, cov
