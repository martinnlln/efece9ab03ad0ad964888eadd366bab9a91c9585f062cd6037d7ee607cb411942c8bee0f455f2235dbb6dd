# ADVANCED QUANTITATIVE PRICE PREDICTION APPLICATION
## Renaissance Technologies-Inspired Multi-Asset Trading System

---

## SYSTEM OVERVIEW

Build a comprehensive quantitative trading platform that predicts prices for CRYPTO, STOCKS, and FOREX using cutting-edge machine learning, sentiment analysis, market microstructure, and multi-session analysis. The system should run distributed across MacBook Pro M2 (16GB RAM) and Desktop PC (Ryzen 5 5600x, RTX 3060 12GB, 32GB RAM upgradable).

---

## CORE ARCHITECTURE

### **Technology Stack**
- **Backend**: Python 3.11+ (FastAPI for API, asyncio for concurrent processing)
- **Frontend**: React + TypeScript, TailwindCSS, Recharts/Plotly for visualization
- **Database**: PostgreSQL (time-series data), Redis (real-time cache)
- **ML Framework**: PyTorch, TensorFlow, scikit-learn
- **Data Processing**: Pandas, NumPy, TA-Lib
- **Message Queue**: RabbitMQ or Redis Streams (for distributed processing)
- **Container**: Docker + Docker Compose

### **Distributed Processing Strategy**
- **MacBook M2**: Real-time data ingestion, API serving, lightweight model inference
- **Desktop PC**: Heavy ML training, batch predictions, deep learning models (GPU acceleration)
- **Communication**: REST API + WebSocket for real-time updates between machines

---

## DATA ACQUISITION (100% FREE APIS)

### **Market Data Sources**

#### **1. Primary APIs (Free Tiers)**
- **Alpha Vantage** (primary): Stocks, Forex, Crypto, 50+ technical indicators
  - 500 requests/day free
  - Real-time + historical data
  - API: `https://www.alphavantage.co/`

- **Finnhub** (secondary): Real-time market data, news sentiment
  - 60 API calls/minute free
  - Social sentiment data
  - API: `https://finnhub.io/`

- **Twelve Data**: Multi-asset coverage
  - 800 requests/day free
  - 100+ technical indicators
  - API: `https://twelvedata.com/`

- **Yahoo Finance** (via yfinance library): Unlimited historical data
  - No API key required
  - Python library: `pip install yfinance`

- **Binance API**: Crypto data (no API key for public endpoints)
  - Real-time OHLCV, order book, trades
  - API: `https://api.binance.com/`

#### **2. Alternative Data**
- **News Sentiment**: Finnhub News API (free tier)
- **Social Sentiment**: 
  - Reddit API (free with account)
  - Twitter API v2 (limited free tier)
  - Financial Modeling Prep Social Sentiment (free tier)

- **Economic Data**: FRED API (Federal Reserve Economic Data - free)
  - Unlimited access, registration required
  - Macro indicators (GDP, unemployment, interest rates)

#### **3. On-Chain Data (Crypto)**
- **CoinGecko API**: Free, unlimited requests
- **Blockchain.info API**: Bitcoin on-chain metrics (free)

---

## RENAISSANCE TECHNOLOGIES STRATEGIES IMPLEMENTATION

### **1. Pattern Recognition & Statistical Arbitrage**

#### **Core Principles**
- Analyze massive datasets for non-random movements
- Short-term pattern exploitation (holding periods: seconds to 2 weeks)
- 50.75% win rate with proper position sizing (Kelly Criterion)

#### **Implementation**
```python
# Hidden Markov Models (HMM) for regime detection
- Use hmmlearn library
- Train on historical data to identify market regimes (trending, ranging, volatile)
- Adjust strategies based on detected regime

# Signal Processing (Speech Recognition Techniques)
- Apply FFT (Fast Fourier Transform) for cycle detection
- Wavelet transforms for multi-resolution analysis
- Auto-correlation analysis for pattern discovery

# Information Theory
- Mutual Information for feature selection
- Entropy-based uncertainty quantification
- Shannon entropy for market efficiency analysis
```

### **2. Machine Learning Models**

#### **Model Stack (Progressive Complexity)**

##### **Tier 1: Traditional ML (Fast Training - MacBook)**
```python
# Gradient Boosting Machines
- XGBoost: Fast, handles missing data
- LightGBM: Memory efficient
- CatBoost: Categorical features handling

# Support Vector Machines (SVM)
- Classification for direction prediction
- Regression for price targets

# Random Forest
- Feature importance extraction
- Ensemble predictions

# Logistic Regression
- Quick baseline model
- Feature correlation analysis
```

##### **Tier 2: Deep Learning (GPU Training - Desktop)**
```python
# Recurrent Networks
- LSTM (Long Short-Term Memory): Time-series sequences
- GRU (Gated Recurrent Units): Faster alternative to LSTM
- Bidirectional LSTM: Forward + backward context

# Transformer Architecture
- Temporal Fusion Transformer (TFT): Multi-horizon forecasting
- Attention mechanisms: Identify important time steps
- Self-attention for factor effectiveness

# Convolutional Networks
- 1D CNN: Pattern recognition in time series
- TCN (Temporal Convolutional Networks): Long sequences

# Hybrid Models
- CNN-LSTM: Feature extraction + sequence modeling
- Transformer + LSTM: Attention + memory
```

##### **Tier 3: Advanced Architectures (Desktop GPU)**
```python
# Reinforcement Learning
- TD3 (Twin Delayed DDPG): Continuous action space
- PPO (Proximal Policy Optimization): Stable training
- DQN: Discrete actions (buy/sell/hold)

# Graph Neural Networks
- Stock correlation graphs
- Market relationship modeling
- Information flow analysis

# Large Language Models (For Sentiment)
- FinBERT: Financial text classification
- FinGPT: Sentiment scoring from news
- Use pre-trained models (no training needed)
```

### **3. Factor Engineering & Alpha Discovery**

#### **200+ Custom Factors (Alpha101 Extended)**

##### **Price-Based Factors**
```python
# Momentum Factors
- RSI variations (2-period to 200-period)
- MACD family (standard, histogram, signal crosses)
- ROC (Rate of Change) multiple timeframes
- Stochastic oscillator variations
- Williams %R
- CCI (Commodity Channel Index)

# Mean Reversion
- Bollinger Band positions
- Z-score calculations
- Distance from moving averages (SMA, EMA, WMA)
- Keltner Channel positions

# Trend Strength
- ADX (Average Directional Index)
- Aroon indicators
- Parabolic SAR
- Ichimoku Cloud components
- SuperTrend indicator
```

##### **Volume-Based Factors**
```python
# Volume Analysis
- OBV (On-Balance Volume)
- Accumulation/Distribution Line
- Chaikin Money Flow
- Volume Rate of Change
- VWAP (Volume Weighted Average Price) distances
- Money Flow Index (MFI)

# Order Flow Indicators
- Bid-ask spread analysis
- Order book imbalance
- Trade size distribution
- Buy/sell volume ratio
```

##### **Volatility Factors**
```python
# Volatility Measures
- ATR (Average True Range) multi-period
- Historical volatility (realized vol)
- Parkinson volatility (high-low range)
- Garman-Klass volatility
- Rogers-Satchell volatility

# Volatility Clustering
- GARCH model outputs
- ARCH effects detection
```

##### **Cross-Asset Factors**
```python
# Correlation Features
- Rolling correlations with SPY, BTC, DXY
- Correlation regime changes
- Beta calculations
- Factor exposures (market, size, value, momentum)

# Relative Strength
- Relative performance vs market
- Sector relative strength
- Cross-market arbitrage signals
```

##### **Alternative Data Factors**
```python
# Sentiment Scores
- News sentiment (VADER, FinBERT)
- Social media sentiment (Reddit, Twitter)
- News volume and controversy
- Sentiment momentum and acceleration

# Time-Based Features
- Day of week effects
- Month/quarter patterns
- Time-to-earnings
- Seasonal patterns
- Holiday effects
```

### **4. Market Microstructure & Session Analysis**

#### **Trading Session Framework**

##### **Session Definitions (UTC)**
```python
SESSIONS = {
    'SYDNEY': {'open': '22:00', 'close': '07:00'},
    'TOKYO': {'open': '00:00', 'close': '09:00'},
    'LONDON': {'open': '08:00', 'close': '16:00'},
    'NEW_YORK': {'open': '13:00', 'close': '22:00'}
}

# Overlap Periods (Highest Liquidity)
OVERLAPS = {
    'TOKYO_LONDON': ('08:00', '09:00'),
    'LONDON_NY': ('13:00', '16:00')  # MOST ACTIVE
}
```

##### **Session-Specific Features**
```python
# For Each Session Extract:
- Opening/closing prices
- High/Low ranges
- Volume profiles
- Volatility patterns
- Liquidity changes
- Price momentum direction
- Gap analysis (close to next open)

# Inter-Session Learning
- How Tokyo session affects London
- London momentum continuation in NY
- NY close impact on Asian open
- Gap trading strategies
- Session handoff patterns
```

##### **Session-Aware Model Training**
```python
# Time-Aware Features
session_features = [
    'current_session',          # One-hot encoded
    'time_in_session',          # Minutes since session open
    'prev_session_return',      # Previous session P&L
    'session_volatility_ratio', # Current vs average
    'overlap_indicator',        # In overlap period?
    'liquidity_score',          # Relative liquidity
    'session_volume_ratio'      # Volume vs session average
]

# Model learns:
1. When to trade aggressively (high liquidity overlaps)
2. When to avoid (low liquidity periods)
3. Session-specific patterns
4. Cross-session momentum transfer
```

---

## MODEL TRAINING PIPELINE

### **Data Preparation**
```python
# 1. Data Collection (Run on MacBook - lightweight)
- Fetch 5+ years historical data
- Store in PostgreSQL (time-series optimized)
- Real-time updates via scheduled jobs

# 2. Feature Engineering (Both machines)
- Calculate 200+ technical indicators
- Generate lag features (1-50 periods)
- Cross-sectional features (sector, market cap)
- Session-based features
- Sentiment scores

# 3. Data Cleaning
- Handle missing values (forward fill, interpolation)
- Remove outliers (IQR method, Z-score)
- Normalize/standardize (MinMaxScaler, StandardScaler)
- Check for look-ahead bias
```

### **Training Strategy**

#### **Walk-Forward Optimization**
```python
# Time-Series Split (No Shuffle!)
train_periods = [
    ('2018-01-01', '2021-12-31'),  # Training
    ('2022-01-01', '2022-12-31'),  # Validation
    ('2023-01-01', '2024-10-16')   # Out-of-sample test
]

# Rolling Window Approach
- Retrain models monthly
- Keep last 3 years of data
- Validate on next month
- Walk forward continuously
```

#### **Ensemble Strategy (Renaissance Approach)**
```python
# Meta-Ensemble Architecture
models = {
    'fast_models': [XGBoost, LightGBM, RandomForest],
    'deep_models': [LSTM, Transformer, CNN-LSTM],
    'rl_models': [TD3, PPO]
}

# Ensemble Methods
1. Simple averaging
2. Weighted by recent performance
3. Meta-learner (StackingRegressor)
4. Bayesian Model Averaging

# Portfolio of Models
- Momentum models (trending markets)
- Mean reversion (ranging markets)
- Breakout models (volatility expansion)
- Switch based on regime detection
```

### **Risk Management (Kelly Criterion)**
```python
# Position Sizing
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    f* = (p * b - q) / b
    where:
    p = win rate
    q = 1 - p
    b = avg_win / avg_loss
    """
    q = 1 - win_rate
    b = avg_win / avg_loss
    kelly_fraction = (win_rate * b - q) / b
    
    # Use 25% of Kelly (safer)
    return kelly_fraction * 0.25

# Dynamic Leverage (like Medallion)
- Base: 5x leverage
- Max: 12.5x (high confidence trades)
- Reduce in high volatility
- Scale by session liquidity
```

---

## SENTIMENT ANALYSIS PIPELINE

### **News Sentiment (Real-Time)**
```python
# 1. News Collection
- Finnhub News API
- Alpha Vantage News Sentiment
- FRED Economic News

# 2. Sentiment Models (Pre-trained - No Training!)
from transformers import pipeline

# Financial BERT
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

# General Sentiment
vader = SentimentIntensityAnalyzer()  # NLTK VADER

# Process Pipeline
def analyze_news(ticker, lookback_hours=24):
    news = fetch_news(ticker, lookback_hours)
    
    sentiments = []
    for article in news:
        # FinBERT score
        finbert_score = finbert(article['headline'])[0]
        
        # VADER score
        vader_score = vader.polarity_scores(article['content'])
        
        # Weighted composite
        composite = (
            finbert_score['score'] * 0.7 +
            vader_score['compound'] * 0.3
        )
        
        sentiments.append({
            'timestamp': article['time'],
            'sentiment': composite,
            'label': finbert_score['label']
        })
    
    return aggregate_sentiment(sentiments)
```

### **Social Media Sentiment**
```python
# Reddit Sentiment (PRAW library)
def reddit_sentiment(ticker, subreddit='wallstreetbets'):
    reddit = praw.Reddit(...)  # Free with account
    
    submissions = reddit.subreddit(subreddit).search(
        ticker, time_filter='day', limit=100
    )
    
    sentiments = []
    for submission in submissions:
        text = submission.title + " " + submission.selftext
        score = analyze_sentiment(text)
        sentiments.append(score)
    
    return np.mean(sentiments)

# Twitter Sentiment (tweepy - limited free tier)
# Aggregate: upvotes, comment sentiment, volume
```

### **Sentiment Features**
```python
sentiment_features = [
    'news_sentiment_1h',
    'news_sentiment_4h',
    'news_sentiment_24h',
    'news_volume',           # Number of articles
    'sentiment_change',      # Momentum
    'sentiment_volatility',  # Disagreement measure
    'social_sentiment',
    'social_volume',
    'controversy_score'      # Conflicting sentiments
]
```

---

## FRONTEND DASHBOARD SPECIFICATIONS

### **Technology**
- React 18 + TypeScript
- TailwindCSS for styling
- Recharts/Plotly for charts
- WebSocket for real-time updates
- React Query for data fetching

### **Dashboard Components**

#### **1. Overview Page**
```
┌─────────────────────────────────────────────────┐
│  MARKET OVERVIEW                   [Refresh 🔄] │
├─────────────────────────────────────────────────┤
│  ┌─────────┬─────────┬─────────┬─────────┐    │
│  │ CRYPTO  │ STOCKS  │ FOREX   │ INDICES │    │
│  │ BTC     │ SPY     │ EUR/USD │ S&P 500 │    │
│  │ +2.4%   │ -0.8%   │ +0.3%   │ -0.8%   │    │
│  │ 🟢 BUY  │ 🔴 SELL │ 🟡 HOLD │ 🔴 SELL│    │
│  └─────────┴─────────┴─────────┴─────────┘    │
│                                                  │
│  LIVE SESSION INDICATOR                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━         │
│  [TOKYO] ━━━━━━━ [LONDON] ━━━ [NY]             │
│          (Closed)  (ACTIVE) (Opening in 2h)     │
│                                                  │
│  TOP PREDICTIONS (Next 24h)                      │
│  1. BTC/USDT: +3.2% (Confidence: 78%)           │
│  2. AAPL: +1.8% (Confidence: 65%)               │
│  3. EUR/USD: -0.5% (Confidence: 72%)            │
└─────────────────────────────────────────────────┘
```

#### **2. Asset Detail Page**
```
┌─────────────────────────────────────────────────┐
│  BTC/USDT                          [1D] [1W] [1M]│
├─────────────────────────────────────────────────┤
│  PRICE CHART with ML PREDICTIONS                 │
│  ┌──────────────────────────────────────────┐  │
│  │     📊 Historical + Predicted (shaded)   │  │
│  │  70k │     ╱╲                            │  │
│  │  65k │   ╱    ╲  ╱╲    ┌─────┐          │  │
│  │  60k │ ╱        ╲    ╲  │ ML  │          │  │
│  │  55k │                ╲ │Pred │          │  │
│  │      └──────────────────┴─────┴─────────┘  │
│  │       Jan  Feb  Mar  Apr  May  Future→   │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  TECHNICAL INDICATORS (Collapsible)              │
│  ▼ Momentum                                      │
│    RSI(14): 68 (Overbought warning)             │
│    MACD: Bullish crossover (2 days ago)         │
│  ▼ Volatility                                    │
│    ATR(14): 2.3% (High)                         │
│    Bollinger: Near upper band                    │
│                                                  │
│  SENTIMENT ANALYSIS                              │
│  News: 🟢 78% Positive (124 articles)            │
│  Social: 🟡 52% Positive (Mixed signals)         │
│  Fear/Greed: 72 (Greed territory)               │
│                                                  │
│  SESSION ANALYSIS                                │
│  Best time to trade: London-NY overlap           │
│  Avg hourly volatility: 1.2%                    │
│  Liquidity score: 9.2/10                        │
│                                                  │
│  ML MODEL PREDICTIONS                            │
│  ┌────────┬──────────┬────────────┬──────────┐ │
│  │ Model  │ 1h Pred  │ 24h Pred   │ Conf %   │ │
│  ├────────┼──────────┼────────────┼──────────┤ │
│  │ LSTM   │ +0.8%    │ +2.1%      │ 73%      │ │
│  │ XGBoost│ +0.5%    │ +1.8%      │ 68%      │ │
│  │ Trans. │ +1.1%    │ +3.2%      │ 81%      │ │
│  │ Ensemb │ +0.9%    │ +2.4%      │ 76%      │ │
│  └────────┴──────────┴────────────┴──────────┘ │
│                                                  │
│  RISK METRICS                                    │
│  Sharpe Ratio: 1.8  │  Max Drawdown: -8%        │
│  Win Rate: 62%      │  Profit Factor: 1.7       │
└─────────────────────────────────────────────────┘
```

#### **3. Portfolio Tracker**
```
┌─────────────────────────────────────────────────┐
│  PORTFOLIO PERFORMANCE                           │
├─────────────────────────────────────────────────┤
│  Total Value: $125,430                           │
│  Today P&L: +$1,234 (+0.98%) 🟢                 │
│  Week: +$4,567 (+3.6%)                          │
│                                                  │
│  ACTIVE POSITIONS                                │
│  ┌─────────┬──────┬────────┬─────────┬──────┐  │
│  │ Asset   │ Size │ Entry  │ Current │ P&L  │  │
│  ├─────────┼──────┼────────┼─────────┼──────┤  │
│  │ BTC     │ 2.5  │ 58000  │ 65000   │+12%  │  │
│  │ AAPL    │ 100  │ 180    │ 185     │+2.8% │  │
│  │ EUR/USD │10000 │ 1.0850 │ 1.0920  │+0.6% │  │
│  └─────────┴──────┴────────┴─────────┴──────┘  │
│                                                  │
│  SUGGESTED ACTIONS (AI-Powered)                  │
│  🔴 Take profit on BTC (target reached)          │
│  🟢 Add to EUR/USD (breakout confirmed)          │
│  🟡 Monitor AAPL (approaching resistance)        │
└─────────────────────────────────────────────────┘
```

#### **4. Model Performance Monitor**
```
┌─────────────────────────────────────────────────┐
│  MODEL PERFORMANCE ANALYTICS                     │
├─────────────────────────────────────────────────┤
│  ACCURACY OVER TIME                              │
│  ┌──────────────────────────────────────────┐  │
│  │ 85%│     ─────                           │  │
│  │ 80%│   /       ─────                     │  │
│  │ 75%│ /               ───                 │  │
│  │ 70%│                     ─────           │  │
│  │    └──────────────────────────────────── │  │
│  │     Jan  Feb  Mar  Apr  May              │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  MODEL COMPARISON                                │
│  ┌──────────┬──────┬──────────┬────────────┐   │
│  │ Model    │ Acc  │ Sharpe   │ Last Update│   │
│  ├──────────┼──────┼──────────┼────────────┤   │
│  │ LSTM     │ 78%  │ 1.9      │ 2h ago     │   │
│  │ XGBoost  │ 72%  │ 1.6      │ 1h ago     │   │
│  │ Transform│ 81%  │ 2.1      │ 30m ago    │   │
│  │ Ensemble │ 83%  │ 2.3      │ Real-time  │   │
│  └──────────┴──────┴──────────┴────────────┘   │
│                                                  │
│  TRAINING STATUS                                 │
│  Desktop PC: Training LSTM (Epoch 45/100)       │
│  MacBook: Serving predictions (240 req/min)     │
│  GPU Utilization: 87%                            │
└─────────────────────────────────────────────────┘
```

---

## DEPLOYMENT ARCHITECTURE

### **Docker Compose Setup**
```yaml
version: '3.8'
services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  # Backend API (MacBook)
  api:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://trader:password@postgres:5432/trading_db
      REDIS_URL: redis://redis:6379
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
  
  # ML Training Service (Desktop PC - GPU)
  ml_trainer:
    build: ./ml_service
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: 0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  # Frontend
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - api
```

### **Communication Flow**
```
┌──────────────┐          ┌──────────────┐
│  MacBook M2  │          │ Desktop PC   │
│              │          │              │
│ - API Server │◄────────►│ - ML Trainer │
│ - Redis Cache│  REST/WS │ - GPU Models │
│ - Postgres   │          │ - Batch Pred │
│ - Data Fetch │          │              │
└──────────────┘          └──────────────┘
       ▲                         ▲
       │                         │
       │    ┌──────────────┐    │
       └────┤   Frontend   ├────┘
            │   (Browser)  │
            └──────────────┘
```

---

## IMPLEMENTATION ROADMAP

### **Phase 1: Foundation (Weeks 1-2)**
1. Set up Docker environment
2. Configure databases (PostgreSQL, Redis)
3. Implement data collection pipeline (all APIs)
4. Create basic frontend structure

### **Phase 2: Feature Engineering (Weeks 3-4)**
1. Build 200+ technical indicators
2. Implement session analysis
3. Set up sentiment analysis pipeline
4. Create feature storage system

### **Phase 3: ML Models (Weeks 5-8)**
1. Implement baseline models (XGBoost, RF)
2. Train LSTM/GRU networks
3. Build Transformer models
4. Create ensemble framework
5. Implement walk-forward validation

### **Phase 4: Strategy & Risk (Weeks 9-10)**
1. Implement Kelly Criterion position sizing
2. Build regime detection
3. Create portfolio manager
4. Implement risk controls

### **Phase 5: Dashboard & Testing (Weeks 11-12)**
1. Complete frontend components
2. Real-time WebSocket updates
3. Backtesting framework
4. Paper trading integration

### **Phase 6: Production (Week 13+)**
1. System optimization
2. Performance monitoring
3. Continuous retraining pipeline
4. Live trading (if desired)

---

## PERFORMANCE TARGETS

### **Prediction Accuracy Goals**
- Directional accuracy: >60% (Renaissance: 50.75%)
- Sharpe Ratio: >2.0 (risk-adjusted returns)
- Maximum Drawdown: <15%
- Win Rate: >55%
- Profit Factor: >1.5

### **System Performance**
- API Response Time: <200ms
- Real-time predictions: <1 second
- Dashboard load time: <2 seconds
- Model inference: <100ms per asset
- Data pipeline latency: <5 seconds

---

## KEY LIBRARIES & DEPENDENCIES

```txt
# Machine Learning
torch>=2.0.0
tensorflow>=2.12.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
catboost>=1.2.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
ta-lib>=0.4.0

# NLP & Sentiment
transformers>=4.30.0
nltk>=3.8.0
vaderSentiment>=3.3.2

# APIs & Data
yfinance>=0.2.28
requests>=2.31.0
websocket-client>=1.6.0
praw>=7.7.0  # Reddit API
tweepy>=4.14.0  # Twitter API

# Backend
fastapi>=0.100.0
uvicorn>=0.23.0
sqlalchemy>=2.0.0
redis>=4.6.0
celery>=5.3.0  # Task queue

# Frontend
# (npm/yarn packages in package.json)
react>=18.2.0
typescript>=5.0.0
recharts>=2.7.0
socket.io-client>=4.6.0

# DevOps
docker>=24.0.0
docker-compose>=2.20.0
```

---

## CRITICAL SUCCESS FACTORS

### **1. Data Quality**
- Clean, consistent data pipelines
- Proper handling of missing values
- Outlier detection and removal
- No look-ahead bias

### **2. Model Robustness**
- Walk-forward validation (not k-fold!)
- Out-of-sample testing
- Regular retraining
- Ensemble diversity

### **3. Risk Management**
- Position sizing (Kelly Criterion)
- Stop losses
- Maximum position limits
- Correlation limits

### **4. Computational Efficiency**
- Optimize feature calculation
- GPU acceleration where possible
- Caching frequently used data
- Parallel processing across machines
- Batch predictions

### **5. Monitoring & Adaptation**
- Track model performance daily
- Detect regime changes
- Automatic retraining triggers
- Alert system for anomalies

---

## DETAILED CODE ARCHITECTURE

### **Backend Structure**
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models
│   │   ├── schemas.py          # Pydantic models
│   ├── api/
│   │   ├── routes/
│   │   │   ├── predictions.py  # Prediction endpoints
│   │   │   ├── market_data.py  # Market data endpoints
│   │   │   ├── portfolio.py    # Portfolio management
│   │   │   ├── models.py       # Model performance endpoints
│   │   ├── websocket.py        # Real-time updates
│   ├── services/
│   │   ├── data_collection/
│   │   │   ├── alpha_vantage.py
│   │   │   ├── finnhub.py
│   │   │   ├── binance.py
│   │   │   ├── yfinance_collector.py
│   │   ├── feature_engineering/
│   │   │   ├── technical_indicators.py
│   │   │   ├── session_features.py
│   │   │   ├── sentiment_features.py
│   │   ├── sentiment/
│   │   │   ├── news_analyzer.py
│   │   │   ├── social_analyzer.py
│   │   │   ├── finbert_model.py
│   │   ├── ml_models/
│   │   │   ├── ensemble.py
│   │   │   ├── model_registry.py
│   │   │   ├── inference.py
│   │   ├── risk/
│   │   │   ├── position_sizing.py
│   │   │   ├── portfolio_manager.py
│   │   │   ├── risk_calculator.py
│   ├── utils/
│   │   ├── cache.py            # Redis caching
│   │   ├── logger.py           # Logging
│   │   ├── validators.py       # Input validation
│   ├── workers/
│   │   ├── data_fetcher.py     # Scheduled data collection
│   │   ├── predictor.py        # Prediction worker
│   │   ├── model_trainer.py    # Retraining worker
├── tests/
├── Dockerfile
└── requirements.txt
```

### **ML Service Structure (Desktop PC)**
```
ml_service/
├── training/
│   ├── train_xgboost.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── train_ensemble.py
│   ├── hyperparameter_tuning.py
├── models/
│   ├── architectures/
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── cnn_lstm.py
│   │   ├── reinforcement/
│   │   │   ├── td3_agent.py
│   │   │   ├── ppo_agent.py
│   ├── saved_models/         # Serialized models
│   ├── checkpoints/          # Training checkpoints
├── evaluation/
│   ├── backtester.py
│   ├── walk_forward.py
│   ├── metrics.py
├── feature_engineering/
│   ├── alpha101.py           # WorldQuant Alpha101 factors
│   ├── custom_factors.py
│   ├── feature_selector.py
├── data/
│   ├── preprocessor.py
│   ├── dataset_builder.py
├── utils/
│   ├── gpu_utils.py
│   ├── model_serializer.py
├── api/
│   ├── model_server.py       # Serves predictions via REST
├── Dockerfile.gpu
└── requirements.txt
```

### **Frontend Structure**
```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard/
│   │   │   ├── Overview.tsx
│   │   │   ├── MarketSessions.tsx
│   │   │   ├── TopPredictions.tsx
│   │   ├── AssetDetail/
│   │   │   ├── PriceChart.tsx
│   │   │   ├── TechnicalIndicators.tsx
│   │   │   ├── SentimentPanel.tsx
│   │   │   ├── PredictionTable.tsx
│   │   ├── Portfolio/
│   │   │   ├── PositionsTable.tsx
│   │   │   ├── PerformanceChart.tsx
│   │   │   ├── SuggestedActions.tsx
│   │   ├── ModelMonitor/
│   │   │   ├── AccuracyChart.tsx
│   │   │   ├── ModelComparison.tsx
│   │   │   ├── TrainingStatus.tsx
│   │   ├── common/
│   │   │   ├── Chart.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── LoadingSpinner.tsx
│   ├── hooks/
│   │   ├── useWebSocket.ts
│   │   ├── usePredictions.ts
│   │   ├── useMarketData.ts
│   ├── services/
│   │   ├── api.ts            # API client
│   │   ├── websocket.ts      # WebSocket connection
│   ├── types/
│   │   ├── market.ts
│   │   ├── prediction.ts
│   │   ├── portfolio.ts
│   ├── utils/
│   │   ├── formatters.ts
│   │   ├── calculations.ts
│   ├── App.tsx
│   ├── index.tsx
├── public/
├── package.json
└── tsconfig.json
```

---

## EXAMPLE IMPLEMENTATION CODE SNIPPETS

### **1. Data Collection Service**
```python
# backend/app/services/data_collection/alpha_vantage.py
import aiohttp
import asyncio
from typing import Dict, List
from datetime import datetime, timedelta

class AlphaVantageCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # 5 calls per minute free tier
    
    async def fetch_intraday(
        self, 
        symbol: str, 
        interval: str = "5min"
    ) -> Dict:
        """Fetch intraday data with technical indicators"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                data = await response.json()
                await asyncio.sleep(self.rate_limit_delay)
                return self._parse_timeseries(data)
    
    async def fetch_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "daily",
        time_period: int = 14
    ) -> Dict:
        """Fetch pre-calculated technical indicators"""
        params = {
            "function": indicator,  # e.g., RSI, MACD, BBANDS
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            "apikey": self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                data = await response.json()
                await asyncio.sleep(self.rate_limit_delay)
                return data
    
    async def fetch_news_sentiment(self, tickers: List[str]) -> Dict:
        """Fetch news sentiment analysis"""
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(tickers),
            "apikey": self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                return await response.json()
    
    def _parse_timeseries(self, data: Dict) -> Dict:
        """Parse and normalize time series data"""
        # Implementation details
        pass
```

### **2. Feature Engineering - Technical Indicators**
```python
# backend/app/services/feature_engineering/technical_indicators.py
import pandas as pd
import numpy as np
import talib

class TechnicalFeatureEngine:
    """Generate 200+ technical indicators"""
    
    def __init__(self):
        self.features = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        Expects df with: open, high, low, close, volume
        """
        df = df.copy()
        
        # Price-based features
        df = self._momentum_indicators(df)
        df = self._trend_indicators(df)
        df = self._volatility_indicators(df)
        df = self._volume_indicators(df)
        
        # Advanced features
        df = self._statistical_features(df)
        df = self._pattern_recognition(df)
        df = self._custom_factors(df)
        
        return df
    
    def _momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI family (multiple periods)
        for period in [2, 7, 14, 21, 50]:
            df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
        
        # MACD variations
        for fast, slow, signal in [(12,26,9), (5,35,5), (8,17,9)]:
            macd, signal_line, hist = talib.MACD(
                close, 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal_line
            df[f'macd_hist_{fast}_{slow}'] = hist
        
        # Stochastic
        for period in [5, 14, 21]:
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=period,
                slowk_period=3,
                slowd_period=3
            )
            df[f'stoch_k_{period}'] = slowk
            df[f'stoch_d_{period}'] = slowd
        
        # ROC (Rate of Change)
        for period in [1, 5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)
        
        # Williams %R
        for period in [14, 21]:
            df[f'willr_{period}'] = talib.WILLR(high, low, close, timeperiod=period)
        
        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            df[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)
        
        # MFI (Money Flow Index)
        df['mfi_14'] = talib.MFI(high, low, close, df['volume'], timeperiod=14)
        
        return df
    
    def _trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Moving averages (multiple types and periods)
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            df[f'wma_{period}'] = talib.WMA(close, timeperiod=period)
            
            # Distance from MA
            df[f'dist_sma_{period}'] = (close - df[f'sma_{period}']) / df[f'sma_{period}']
        
        # ADX (Average Directional Index)
        for period in [14, 21]:
            df[f'adx_{period}'] = talib.ADX(high, low, close, timeperiod=period)
            df[f'plus_di_{period}'] = talib.PLUS_DI(high, low, close, timeperiod=period)
            df[f'minus_di_{period}'] = talib.MINUS_DI(high, low, close, timeperiod=period)
        
        # Aroon
        aroon_down, aroon_up = talib.AROON(high, low, timeperiod=25)
        df['aroon_down'] = aroon_down
        df['aroon_up'] = aroon_up
        df['aroon_osc'] = aroon_up - aroon_down
        
        # Parabolic SAR
        df['sar'] = talib.SAR(high, low)
        
        return df
    
    def _volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(
                close, 
                timeperiod=period,
                nbdevup=2,
                nbdevdn=2
            )
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (close - lower) / (upper - lower)
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = np.log(close / close.shift(1))
            df[f'hist_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252)
        
        # Keltner Channels
        for period in [20]:
            ema = talib.EMA(close, timeperiod=period)
            atr = talib.ATR(high, low, close, timeperiod=period)
            df[f'keltner_upper_{period}'] = ema + (2 * atr)
            df[f'keltner_lower_{period}'] = ema - (2 * atr)
        
        return df
    
    def _volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # OBV (On-Balance Volume)
        df['obv'] = talib.OBV(close, volume)
        df['obv_ema'] = talib.EMA(df['obv'], timeperiod=20)
        
        # Accumulation/Distribution
        df['ad'] = talib.AD(high, low, close, volume)
        df['adosc'] = talib.ADOSC(high, low, close, volume)
        
        # Chaikin Money Flow
        for period in [20]:
            mfm = ((close - low) - (high - close)) / (high - low)
            mfv = mfm * volume
            df[f'cmf_{period}'] = mfv.rolling(period).sum() / volume.rolling(period).sum()
        
        # Volume Rate of Change
        for period in [5, 10]:
            df[f'vroc_{period}'] = talib.ROC(volume, timeperiod=period)
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
        df['dist_vwap'] = (close - df['vwap']) / df['vwap']
        
        return df
    
    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features"""
        close = df['close']
        
        # Z-scores
        for period in [20, 50]:
            mean = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'zscore_{period}'] = (close - mean) / std
        
        # Skewness and Kurtosis
        for period in [20]:
            returns = close.pct_change()
            df[f'skew_{period}'] = returns.rolling(period).skew()
            df[f'kurt_{period}'] = returns.rolling(period).kurt()
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'autocorr_{lag}'] = close.rolling(20).apply(
                lambda x: pd.Series(x).autocorr(lag=lag)
            )
        
        return df
    
    def _pattern_recognition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern recognition"""
        open_price = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Major candlestick patterns
        patterns = [
            'CDLDOJI', 'CDLHAMMER', 'CDLINVERTEDHAMMER',
            'CDLENGULFING', 'CDLHARAMI', 'CDLPIERCING',
            'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR',
            'CDLMARUBOZU', 'CDLSPINNINGTOP'
        ]
        
        for pattern in patterns:
            func = getattr(talib, pattern)
            df[pattern.lower()] = func(open_price, high, low, close)
        
        return df
    
    def _custom_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom alpha factors inspired by WorldQuant Alpha101"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Alpha #1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
        returns = close.pct_change()
        stddev_20 = returns.rolling(20).std()
        condition = returns < 0
        power_base = np.where(condition, stddev_20, close)
        df['alpha1'] = power_base ** 2
        
        # Alpha #2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
        log_vol_delta = np.log(volume).diff(2)
        pct_change = (close - df['open']) / df['open']
        df['alpha2'] = -1 * log_vol_delta.rolling(6).corr(pct_change)
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'price_mom_{period}'] = close / close.shift(period) - 1
        
        # Volume momentum
        for period in [5, 10]:
            df[f'vol_mom_{period}'] = volume / volume.shift(period) - 1
        
        # High-Low ratio
        df['hl_ratio'] = high / low
        df['hl_ratio_ma'] = df['hl_ratio'].rolling(20).mean()
        
        return df
```

### **3. Session Feature Engineering**
```python
# backend/app/services/feature_engineering/session_features.py
import pandas as pd
from datetime import datetime, time
from typing import Dict

class SessionFeatureEngine:
    """Extract market session-specific features"""
    
    SESSIONS = {
        'SYDNEY': {'open': time(22, 0), 'close': time(7, 0)},
        'TOKYO': {'open': time(0, 0), 'close': time(9, 0)},
        'LONDON': {'open': time(8, 0), 'close': time(16, 0)},
        'NEW_YORK': {'open': time(13, 0), 'close': time(22, 0)}
    }
    
    OVERLAPS = {
        'TOKYO_LONDON': (time(8, 0), time(9, 0)),
        'LONDON_NY': (time(13, 0), time(16, 0))
    }
    
    def calculate_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all session-related features"""
        df = df.copy()
        
        # Identify current session
        df['current_session'] = df.index.time.map(self._get_session)
        
        # Session-specific statistics
        df = self._calculate_session_stats(df)
        
        # Inter-session features
        df = self._calculate_inter_session_features(df)
        
        # Overlap indicators
        df = self._calculate_overlap_features(df)
        
        # Time-based features
        df = self._calculate_time_features(df)
        
        return df
    
    def _get_session(self, timestamp_time: time) -> str:
        """Determine which session a timestamp belongs to"""
        for session_name, times in self.SESSIONS.items():
            if self._is_in_session(timestamp_time, times['open'], times['close']):
                return session_name
        return 'CLOSED'
    
    def _is_in_session(self, t: time, open_t: time, close_t: time) -> bool:
        """Check if time is within session (handles overnight sessions)"""
        if open_t < close_t:
            return open_t <= t < close_t
        else:  # Overnight session (e.g., Sydney)
            return t >= open_t or t < close_t
    
    def _calculate_session_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistics for each session"""
        # Group by date and session
        df['date'] = df.index.date
        
        session_stats = df.groupby(['date', 'current_session']).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()
        
        # Calculate session returns
        session_stats['session_return'] = (
            (session_stats['close'] - session_stats['open']) / 
            session_stats['open']
        )
        
        # Merge back to main dataframe
        df = df.merge(
            session_stats[['date', 'current_session', 'session_return']],
            on=['date', 'current_session'],
            how='left'
        )
        
        # Previous session return
        df['prev_session_return'] = df.groupby('current_session')['session_return'].shift(1)
        
        # Session volatility (ATR)
        session_atr = (session_stats['high'] - session_stats['low']).groupby(
            session_stats['current_session']
        ).mean()
        
        df['session_avg_volatility'] = df['current_session'].map(session_atr)
        df['session_volatility_ratio'] = (df['high'] - df['low']) / df['session_avg_volatility']
        
        return df
    
    def _calculate_inter_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features showing how sessions affect each other"""
        # Gap between sessions
        df['session_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Tokyo -> London momentum transfer
        tokyo_close = df[df['current_session'] == 'TOKYO'].groupby('date')['close'].last()
        london_open = df[df['current_session'] == 'LONDON'].groupby('date')['open'].first()
        tokyo_to_london = pd.DataFrame({
            'tokyo_london_gap': (london_open - tokyo_close) / tokyo_close
        })
        df = df.merge(tokyo_to_london, left_on='date', right_index=True, how='left')
        
        # London -> NY momentum transfer
        london_close = df[df['current_session'] == 'LONDON'].groupby('date')['close'].last()
        ny_open = df[df['current_session'] == 'NEW_YORK'].groupby('date')['open'].first()
        london_to_ny = pd.DataFrame({
            'london_ny_gap': (ny_open - london_close) / london_close
        })
        df = df.merge(london_to_ny, left_on='date', right_index=True, how='left')
        
        return df
    
    def _calculate_overlap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for session overlap periods"""
        df['in_overlap'] = df.index.time.map(self._is_overlap)
        df['overlap_name'] = df.index.time.map(self._get_overlap_name)
        
        # Liquidity score (higher during overlaps)
        df['liquidity_score'] = df['in_overlap'].map({True: 10, False: 5})
        
        # Average volume during overlaps
        overlap_avg_volume = df[df['in_overlap']].groupby('overlap_name')['volume'].mean()
        df['overlap_volume_ratio'] = df['volume'] / df['overlap_name'].map(overlap_avg_volume)
        
        return df
    
    def _is_overlap(self, timestamp_time: time) -> bool:
        """Check if time is in an overlap period"""
        for (start, end) in self.OVERLAPS.values():
            if start <= timestamp_time < end:
                return True
        return False
    
    def _get_overlap_name(self, timestamp_time: time) -> str:
        """Get overlap name if in overlap, else None"""
        for name, (start, end) in self.OVERLAPS.items():
            if start <= timestamp_time < end:
                return name
        return 'NO_OVERLAP'
    
    def _calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based cyclical features"""
        # Hour of day (cyclical encoding)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (cyclical)
        df['dayofweek'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Month (seasonality)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Time to next major event (earnings, Fed meeting, etc.)
        # This would require external calendar data
        
        return df
```

### **4. LSTM Model Implementation**
```python
# ml_service/models/architectures/lstm.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Tuple

class LSTMPricePredictor(pl.LightningModule):
    """
    Multi-layer LSTM for price prediction with attention
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)  # [price_1h, price_24h, direction]
        )
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            predictions: (batch_size, 3)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(