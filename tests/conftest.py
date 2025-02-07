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
