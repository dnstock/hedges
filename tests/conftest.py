import pytest
import pandas as pd
import numpy as np

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

@pytest.fixture
def sample_market_data():
    """Provides sample market data for RL testing"""
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
