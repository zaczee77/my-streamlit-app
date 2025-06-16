from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    return score['compound']

def mock_news_sentiment(tickers):
    sentiment_scores = {}
    for ticker in tickers:
        text = f"{ticker} has strong earnings this quarter but faces market volatility"
        sentiment_scores[ticker] = get_sentiment(text)
    return sentiment_scores
