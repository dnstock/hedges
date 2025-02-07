import pytest
import pandas as pd
import numpy as np

# NLP Testing Fixtures
@pytest.fixture
def nlp_config():
    """Provides a standard NLP configuration for testing"""
    return {
        "text_data_path": "tests/data/test_news.csv",
        "model_checkpoint": "ProsusAI/finbert",
        "batch_size": 4,
        "max_length": 128,
        "text_column": "headline",
        "buy_threshold": 0.2,
        "sell_threshold": -0.2,
        "output_path": "tests/data/test_output.csv"
    }

@pytest.fixture
def sample_news_data():
    """Provides a small dataset of financial news headlines"""
    return pd.DataFrame({
        'headline': [
            "Company XYZ reports record profits",
            "Market crash: Stocks plummet",
            "Trading remains stable amid uncertainty",
            "Tech stocks surge to new highs",
            "Inflation concerns worry investors"
        ],
        'date': [
            '2023-01-01',
            '2023-01-02',
            '2023-01-03',
            '2023-01-04',
            '2023-01-05'
        ]
    })

# RL Training Fixtures
@pytest.fixture
def rl_training_config():
    """Provides a standard RL training configuration for testing"""
    return {
        "env": {
            "ticker_list": ["AAPL", "MSFT", "GOOG"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "time_interval": "1D"
        },
        "training": {
            "timesteps": 10000,
            "learning_rate": 0.001,
            "batch_size": 64
        },
        "agent": "PPO"
    }

# Common Market Data Fixtures
@pytest.fixture
def sample_market_data():
    """Provides sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    data = []

    for ticker in ['AAPL', 'MSFT', 'GOOG']:
        ticker_data = pd.DataFrame({
            'date': dates,
            'tic': ticker,
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(100, 200, len(dates)),
            'low': np.random.uniform(100, 200, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            # Add technical indicators
            'macd': np.random.uniform(-2, 2, len(dates)),
            'rsi_30': np.random.uniform(0, 100, len(dates)),
            'cci_30': np.random.uniform(-100, 100, len(dates)),
            'dx_30': np.random.uniform(0, 100, len(dates))
        })
        data.append(ticker_data)

    return pd.concat(data, ignore_index=True)

# Backtesting Fixtures
@pytest.fixture
def backtest_config():
    """Provides standard backtesting configuration"""
    return {
        "env": {
            "ticker_list": ["AAPL", "MSFT", "GOOG"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "time_interval": "1D",
            "initial_amount": 1000000,
            "max_shares": 100
        },
        "agent": "PPO",
        "model_filename": "PPO_model.zip"
    }

@pytest.fixture
def sample_portfolio_history():
    """Provides sample portfolio value history for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    initial_value = 1000000

    # Generate realistic-looking portfolio values with some randomness
    returns = np.random.normal(0.0001, 0.02, len(dates))  # Daily returns
    cumulative_returns = (1 + returns).cumprod()
    portfolio_values = initial_value * cumulative_returns

    return pd.DataFrame({
        'date': dates,
        'total_asset': portfolio_values,
        'daily_return': returns,
        'position': np.random.randint(0, 100, len(dates))
    })

@pytest.fixture
def mock_model_output():
    """Provides sample model predictions for testing"""
    return {
        'actions': np.array([0.1, -0.2, 0.0]),  # Buy, Sell, Hold
        'values': np.array([100.0, 95.0, 98.0]),
        'log_probs': np.array([-0.5, -0.8, -0.3])
    }

# Environment Configuration Fixtures
@pytest.fixture
def env_config():
    """Provides standard environment configuration for both training and testing"""
    return {
        "state_space": 15,  # 5 features × 3 stocks
        "action_space": 3,  # Number of stocks
        "tech_indicator_list": ["macd", "rsi_30", "cci_30", "dx_30"],
        "turbulence_threshold": 100,
        "max_shares": 100,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4
    }

# Execution Testing Fixtures
@pytest.fixture
def execution_config():
    """Provides standard execution configuration for testing"""
    return {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "paper_trading": True,
        "base_currency": "USD",
        "initial_cash": 1000000,
        "ticker_list": ["AAPL", "MSFT", "GOOG"],
        "loop_interval_sec": 60,
        "model_filename": "PPO_model.zip",
        "risk_management": {
            "max_position_size": 100,
            "daily_loss_limit": 500.0
        }
    }

@pytest.fixture
def mock_alpaca_bar():
    """Provides a mock Alpaca bar data structure"""
    class MockBar:
        def __init__(self):
            self.t = pd.Timestamp.now(tz="UTC")
            self.o = 150.0
            self.h = 152.0
            self.l = 149.0
            self.c = 151.0
            self.v = 10000
    return MockBar()

@pytest.fixture
def mock_alpaca_quote():
    """Provides a mock Alpaca quote data structure"""
    class MockQuote:
        def __init__(self):
            self.ap = 151.0  # ask price
            self.bp = 150.9  # bid price
            self.t = pd.Timestamp.now(tz="UTC")
    return MockQuote()

@pytest.fixture
def mock_market_prices():
    """Provides sample current market prices"""
    return {
        "AAPL": 150.0,
        "MSFT": 300.0,
        "GOOG": 2500.0
    }
