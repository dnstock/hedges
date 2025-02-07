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
            'volume': np.random.randint(1000000, 5000000, len(dates))
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
