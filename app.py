# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)

# Cache for API data to avoid repeated calls
cache = {}
cache_expiry = {}
CACHE_DURATION = 3600  # 1 hour

def get_crypto_data(symbol, days=30):
    """Fetch cryptocurrency data from CoinGecko API"""
    # Check cache first
    cache_key = f"{symbol}_{days}"
    current_time = datetime.now()
    
    if cache_key in cache and cache_expiry.get(cache_key, datetime.min) > current_time:
        return cache[cache_key]
    
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Process data
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        price_data = []
        for i, (timestamp, price) in enumerate(prices):
            date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
            volume = volumes[i][1] if i < len(volumes) else 0
            price_data.append({
                'date': date,
                'price': price,
                'volume': volume
            })
        
        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate technical indicators
        df['price_change'] = df['price'].pct_change()
        df['price_5d_mean'] = df['price'].rolling(window=5).mean()
        df['price_20d_mean'] = df['price'].rolling(window=20).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['volume_5d_mean'] = df['volume'].rolling(window=5).mean()
        
        # Store in cache
        cache[cache_key] = df
        cache_expiry[cache_key] = current_time + timedelta(seconds=CACHE_DURATION)
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_features(df):
    """Generate features for prediction model"""
    if df.empty:
        return pd.DataFrame()
    
    # Create features
    features = df.copy()
    
    # Add momentum indicators
    features['price_momentum_5d'] = features['price'] / features['price_5d_mean'] - 1
    features['price_momentum_20d'] = features['price'] / features['price_20d_mean'] - 1
    
    # Add crossover signal
    features['ma_crossover'] = (features['price_5d_mean'] > features['price_20d_mean']).astype(int)
    
    # Add volume indicators
    features['volume_momentum'] = features['volume'] / features['volume_5d_mean'] - 1
    
    # Replace infinities and NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()
    
    return features

def predict_trend(df, days_to_predict=7):
    """Predict price trend using a simple Random Forest model"""
    if df.empty or len(df) < 30:  # Need sufficient data
        return "insufficient_data", {}
    
    features = generate_features(df)
    if features.empty:
        return "insufficient_data", {}
    
    # Feature columns
    feature_cols = ['price_momentum_5d', 'price_momentum_20d', 'ma_crossover', 'volume_momentum']
    
    # Prepare data for training
    features['future_price_change'] = features['price'].shift(-days_to_predict) / features['price'] - 1
    features['target'] = (features['future_price_change'] > 0).astype(int)  # 1 if price goes up, 0 if down
    
    # Drop rows with NaN in target
    model_data = features.dropna()
    
    if len(model_data) < 15:  # Need sufficient training examples
        return "insufficient_data", {}
    
    # Split data for training
    X = model_data[feature_cols].values
    y = model_data['target'].values
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Make prediction on the latest data
    latest_features = features[feature_cols].iloc[-1].values.reshape(1, -1)
    prediction_prob = model.predict_proba(latest_features)[0]
    
    # Get prediction class (0=down, 1=up) and probability
    prediction_class = 1 if prediction_prob[1] > 0.5 else 0
    confidence = prediction_prob[1] if prediction_class == 1 else prediction_prob[0]
    
    # Generate recommendation
    if prediction_class == 1 and confidence > 0.65:
        recommendation = "buy"
    elif prediction_class == 0 and confidence > 0.65:
        recommendation = "sell"
    else:
        recommendation = "hold"
    
    # Additional signals
    signals = {}
    signals['ma_crossover'] = "bullish" if features['ma_crossover'].iloc[-1] == 1 else "bearish"
    signals['volume_trend'] = "increasing" if features['volume_change'].iloc[-1] > 0 else "decreasing"
    signals['price_trend_5d'] = "up" if features['price_momentum_5d'].iloc[-1] > 0 else "down"
    signals['price_trend_20d'] = "up" if features['price_momentum_20d'].iloc[-1] > 0 else "down"
    signals['confidence'] = f"{confidence:.2%}"
    
    return recommendation, signals

def create_price_chart(df, symbol):
    """Create price chart with moving averages"""
    if df.empty:
        return {}
    
    # Create traces
    trace_price = go.Scatter(
        x=df['date'],
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#17BECF')
    )
    
    trace_ma5 = go.Scatter(
        x=df['date'],
        y=df['price_5d_mean'],
        mode='lines',
        name='5-Day MA',
        line=dict(color='#7F7F7F')
    )
    
    trace_ma20 = go.Scatter(
        x=df['date'],
        y=df['price_20d_mean'],
        mode='lines',
        name='20-Day MA',
        line=dict(color='#DE3163')
    )
    
    data = [trace_price, trace_ma5, trace_ma20]
    
    layout = go.Layout(
        title=f'{symbol.upper()} Price Chart',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Price (USD)'),
        hovermode='closest'
    )
    
    figure = {'data': data, 'layout': layout}
    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
def index():
    # Default to Bitcoin
    return render_template('index.html', 
                          cryptocurrencies=['bitcoin', 'ethereum', 'cardano', 'solana', 'ripple'])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol', 'bitcoin')
    days = int(data.get('days', 30))
    
    df = get_crypto_data(symbol, days)
    if df.empty:
        return jsonify({'error': 'Failed to fetch data'})
    
    recommendation, signals = predict_trend(df)
    chart_json = create_price_chart(df, symbol)
    
    last_price = df['price'].iloc[-1] if not df.empty else 0
    price_change = df['price_change'].iloc[-1] if not df.empty else 0
    
    return jsonify({
        'recommendation': recommendation,
        'signals': signals,
        'chart': chart_json,
        'last_price': f"${last_price:.2f}",
        'price_change': f"{price_change:.2%}" if price_change else "N/A"
    })

if __name__ == '__main__':
    # Get port from environment variable for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
