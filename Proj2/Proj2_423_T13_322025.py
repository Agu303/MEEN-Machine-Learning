import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob  # for basic sentiment analysis
import matplotlib.pyplot as plt
import yfinance as yf  # for getting Bitcoin price data
import ta  # for technical indicators
import tweepy
import praw
from pytrends.request import TrendReq
import os
from dotenv import load_dotenv

'''
Combination of Sentiment Analysis and LSTM for bitcoin price prediction
1. Price-Value LSTM Model:  Historical data from Kaggle or YFinance datasets, capture temporal dependencies, patterns in price shifts
2. Sentiment:  Twitter/X, Reddit (r/Bitcoin, r/WallStreetBets, r/CryptoCurrency), Google Trends Data
3. Technical Indicators: RSI, MACD, Bollinger Bands, Volume Metrics
'''

# Load environment variables
load_dotenv()

def load_crypto_data(file_path):
    try:
        print("Loading crypto price data...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print("Data loaded successfully! 📈")
        return df
    except FileNotFoundError:
        print("Error: Could not find crypto_price_data.csv")
        exit()
    except Exception as e:
        print("Error loading data:", str(e))
        exit()

def preprocess_data(df):
    print("Preprocessing data...")
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']].values)
    print("Data preprocessing complete! 🔄")
    return scaled_data, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

def load_and_preprocess_price_data(symbol='BTC-USD', period='2y'):
    print("Loading price data...")
    # Get Bitcoin price data
    btc_data = yf.download(symbol, period=period, interval='1d')
    
    # Add technical indicators
    btc_data['RSI'] = ta.momentum.RSIIndicator(btc_data['Close']).rsi()
    btc_data['MACD'] = ta.trend.MACD(btc_data['Close']).macd()
    btc_data['BB_high'] = ta.volatility.BollingerBands(btc_data['Close']).bollinger_hband()
    
    return btc_data

def create_technical_features(df):
    # Create features from technical indicators
    scaler = MinMaxScaler()
    tech_features = scaler.fit_transform(df[['RSI', 'MACD', 'BB_high']].values)
    return tech_features, scaler

def build_hybrid_model(price_input_shape, sentiment_input_shape, tech_input_shape):
    # Price stream (LSTM)
    price_input = Input(shape=price_input_shape)
    price_lstm = LSTM(50, return_sequences=True)(price_input)
    price_lstm = Dropout(0.2)(price_lstm)
    price_lstm = LSTM(50)(price_lstm)
    price_output = Dense(25)(price_lstm)

    # Sentiment stream
    sentiment_input = Input(shape=sentiment_input_shape)
    sentiment_dense = Dense(32, activation='relu')(sentiment_input)
    sentiment_output = Dense(16, activation='relu')(sentiment_dense)

    # Technical indicators stream
    tech_input = Input(shape=tech_input_shape)
    tech_dense = Dense(32, activation='relu')(tech_input)
    tech_output = Dense(16, activation='relu')(tech_dense)

    # Combine all streams
    combined = Concatenate()([price_output, sentiment_output, tech_output])
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    output = Dense(1)(combined)

    model = Model(
        inputs=[price_input, sentiment_input, tech_input],
        outputs=output
    )
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def process_sentiment_data(sentiment_data):
    # Convert sentiment text to numerical values
    sentiments = []
    for text in sentiment_data:
        blob = TextBlob(text)
        sentiments.append([
            blob.sentiment.polarity,
            blob.sentiment.subjectivity
        ])
    return np.array(sentiments)

def collect_twitter_data():
    auth = tweepy.OAuthHandler(
        os.getenv('TWITTER_API_KEY'),
        os.getenv('TWITTER_API_SECRET')
    )
    api = tweepy.API(auth)
    
    # Collect tweets with Bitcoin-related keywords
    bitcoin_tweets = api.search_tweets(
        q="bitcoin OR #btc OR #bitcoin",
        lang="en",
        count=100
    )
    return bitcoin_tweets

def collect_reddit_data():
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent="your_app_name"
    )
    
    subreddits = ['Bitcoin', 'CryptoCurrency', 'wallstreetbets']
    posts = []
    
    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).hot(limit=100):
            posts.append({
                'title': post.title,
                'score': post.score,
                'created_utc': post.created_utc
            })
    
    return posts

def get_google_trends():
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = ["bitcoin", "btc", "crypto"]
    pytrends.build_payload(kw_list, timeframe='today 3-m')
    
    return pytrends.interest_over_time()

def collect_comprehensive_dataset():
    # Dictionary to store all data
    data = {}
    
    # 1. Price and Technical Data
    print("Collecting price data...")
    data['price'] = load_and_preprocess_price_data()
    
    # 2. Social Media Sentiment
    print("Collecting social media data...")
    try:
        data['twitter'] = collect_twitter_data()
    except Exception as e:
        print(f"Twitter data collection failed: {e}")
    
    try:
        data['reddit'] = collect_reddit_data()
    except Exception as e:
        print(f"Reddit data collection failed: {e}")
    
    # 3. Google Trends
    print("Collecting Google Trends data...")
    try:
        data['google_trends'] = get_google_trends()
    except Exception as e:
        print(f"Google Trends collection failed: {e}")
    
    # 4. News Data
    print("Collecting news data...")
    try:
        data['news'] = collect_news_data()
    except Exception as e:
        print(f"News data collection failed: {e}")
    
    return data

def preprocess_comprehensive_data(data):
    processed_data = {}
    
    # 1. Process price data (already handled in your code)
    processed_data['price'] = data['price']
    
    # 2. Process social media sentiment
    social_sentiments = []
    if 'twitter' in data:
        for tweet in data['twitter']:
            sentiment = TextBlob(tweet.text).sentiment
            social_sentiments.append({
                'source': 'twitter',
                'timestamp': tweet.created_at,
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            })
    
    if 'reddit' in data:
        for post in data['reddit']:
            sentiment = TextBlob(post['title']).sentiment
            social_sentiments.append({
                'source': 'reddit',
                'timestamp': post['created_utc'],
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity
            })
    
    processed_data['social_sentiment'] = pd.DataFrame(social_sentiments)
    
    return processed_data

# Load and preprocess data
print("\n Starting crypto price prediction pipeline...")
file_path = 'crypto_price_data.csv'  # Update with actual path
df = load_crypto_data(file_path)
scaled_data, scaler = preprocess_data(df)

# Create sequences
print("Creating sequences for LSTM model...")
seq_length = 60  # Using past 60 days
X, y = create_sequences(scaled_data, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
print("Splitting data into training and testing sets...")
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build and train model
print("\n⏱️ Training LSTM model...")
model = build_lstm_model((X_train.shape[1], 1))
model = train_model(model, X_train, y_train)

# Predict
print("Making predictions...")
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

# Plot results
print("Generating visualization...")
plt.plot(df.index[split + seq_length + 1:], scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual Prices')
plt.plot(df.index[split + seq_length + 1:], predicted_prices, label='Predicted Prices')
plt.legend()
plt.show()
print("\nAnalysis complete! All tasks finished successfully.")

# Example usage:
def train_hybrid_model():
    # 1. Load price data and create technical indicators
    price_data = load_and_preprocess_price_data()
    tech_features, tech_scaler = create_technical_features(price_data)
    
    # 2. Create sequences for LSTM
    seq_length = 60
    price_sequences = create_sequences(price_data['Close'].values, seq_length)
    
    # 3. Load and process sentiment data (you'll need to implement this)
    # sentiment_data = load_sentiment_data()
    # sentiment_features = process_sentiment_data(sentiment_data)
    
    # 4. Build and train hybrid model
    model = build_hybrid_model(
        price_input_shape=(seq_length, 1),
        sentiment_input_shape=(2,),  # polarity and subjectivity
        tech_input_shape=(3,)  # RSI, MACD, BB
    )
    
    # 5. Train model
    # model.fit([price_sequences, sentiment_features, tech_features], y_train,
    #          epochs=10, batch_size=32, validation_split=0.2)
    
    return model
