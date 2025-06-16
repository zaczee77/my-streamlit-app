import sys
import os

# Allow imports from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt

from scripts.optimizer import fetch_price_data, mean_variance_optimization
from scripts.sentiment import mock_news_sentiment

st.title("📈 Smart Portfolio Optimizer with Sentiment Overlay")

# Sidebar inputs
st.sidebar.header("Portfolio Configuration")
tickers = st.sidebar.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN").split(',')
start_date = st.sidebar.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Get data
prices = fetch_price_data(tickers, str(start_date), str(end_date))

if prices.empty:
    st.error("No valid stock data found. Please check the tickers and date range.")
    st.stop()

# Optimization
weights, mu, cov = mean_variance_optimization(prices)

if sum(weights) == 0 or any(pd.isna(weights)):
    st.error("Optimization failed. Try a different date range or tickers.")
    st.stop()

# Sentiment
sentiment = mock_news_sentiment(tickers)

# Adjust weights based on sentiment safely
adjusted_weights = []
for i, ticker in enumerate(tickers):
    sentiment_score = sentiment.get(ticker, 0)
    adjusted = weights[i] * (1 + sentiment_score)
    adjusted_weights.append(max(adjusted, 0))  # ensure non-negative

total = sum(adjusted_weights)
if total == 0:
    st.warning("Sentiment adjustment zeroed out all weights. Reverting to original weights.")
    adjusted_weights = weights
    total = sum(adjusted_weights)

adjusted_weights = [w / total for w in adjusted_weights]

# Debug output (optional)
# st.write("Original Weights:", weights)
# st.write("Sentiment Scores:", sentiment)
# st.write("Adjusted Weights:", adjusted_weights)

# Show table
st.subheader("Adjusted Portfolio Allocation")
df = pd.DataFrame({"Ticker": tickers, "Adjusted Weight": adjusted_weights})
st.dataframe(df.set_index("Ticker"))


fig, ax = plt.subplots()
ax.pie(adjusted_weights, labels=tickers, autopct='%1.1f%%')
ax.axis("equal")
st.pyplot(fig)
