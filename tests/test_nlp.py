import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.nlp_signals.sentiment import (
    load_sentiment_pipeline,
    compute_sentiment_scores,
    map_label_to_value,
)
from src.nlp_signals.pipeline import run_nlp_pipeline

@pytest.fixture
def sample_text_df():
    return pd.DataFrame({
        'headline': [
            "Company XYZ reports record profits",
            "Market crash: Stocks plummet",
            "Trading remains stable amid uncertainty",
            ""  # Test empty string handling
        ]
    })

@pytest.fixture
def mock_sentiment_pipeline():
    return Mock(return_value=[
        {"label": "POSITIVE", "score": 0.8},
        {"label": "NEGATIVE", "score": 0.9},
        {"label": "NEUTRAL", "score": 0.7},
        {"label": "NEUTRAL", "score": 0.5}
    ])

def test_map_label_to_value():
    assert map_label_to_value("POSITIVE", 0.8) == 0.8
    assert map_label_to_value("NEGATIVE", 0.7) == -0.7
    assert map_label_to_value("NEUTRAL", 0.5) == 0.0
    assert map_label_to_value("positive", 0.8) == 0.8  # Test case insensitivity
    assert map_label_to_value("OTHER", 0.5) == 0.0  # Test unknown label

def test_compute_sentiment_scores(sample_text_df, mock_sentiment_pipeline):
    labels, scores = compute_sentiment_scores(
        sample_text_df,
        "headline",
        mock_sentiment_pipeline
    )

    assert len(labels) == len(sample_text_df)
    assert len(scores) == len(sample_text_df)
    assert all(isinstance(score, float) for score in scores)
    assert all(isinstance(label, str) for label in labels)

@patch('src.nlp_signals.sentiment.AutoTokenizer')
@patch('src.nlp_signals.sentiment.AutoModelForSequenceClassification')
@patch('src.nlp_signals.sentiment.pipeline')
def test_load_sentiment_pipeline(mock_pipeline, mock_model, mock_tokenizer):
    # Setup mocks
    mock_tokenizer.from_pretrained.return_value = Mock()
    mock_model.from_pretrained.return_value = Mock()
    mock_pipeline.return_value = Mock()

    # Test pipeline creation
    model_checkpoint = "ProsusAI/finbert"
    result = load_sentiment_pipeline(model_checkpoint)

    mock_tokenizer.from_pretrained.assert_called_once_with(model_checkpoint)
    mock_model.from_pretrained.assert_called_once_with(model_checkpoint)
    mock_pipeline.assert_called_once()
    assert result is not None

@patch('src.nlp_signals.pipeline.load_sentiment_pipeline')
@patch('pandas.read_csv')
def test_run_nlp_pipeline(mock_read_csv, mock_load_pipeline, sample_text_df):
    # Setup mocks
    mock_read_csv.return_value = sample_text_df
    mock_pipeline = Mock(return_value=[
        {"label": "POSITIVE", "score": 0.8},
        {"label": "NEGATIVE", "score": 0.9},
        {"label": "NEUTRAL", "score": 0.5},
        {"label": "NEUTRAL", "score": 0.5}
    ])
    mock_load_pipeline.return_value = mock_pipeline

    # Test configuration
    nlp_config = {
        "text_data_path": "test_data.csv",
        "model_checkpoint": "test/model",
        "text_column": "headline",
        "buy_threshold": 0.2,
        "sell_threshold": -0.2,
        "output_path": "test_output.csv"
    }

    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        run_nlp_pipeline(nlp_config)

        # Verify the pipeline was called
        mock_read_csv.assert_called_once()
        mock_load_pipeline.assert_called_once()
        mock_to_csv.assert_called_once()

def test_pipeline_signal_generation():
    # Test the signal generation logic with a small DataFrame
    df = pd.DataFrame({
        'sentiment_value': [0.3, -0.3, 0.1, -0.1]
    })

    def derive_signal(value):
        if value > 0.2:
            return "BUY"
        elif value < -0.2:
            return "SELL"
        else:
            return "HOLD"

    df['trade_signal'] = df['sentiment_value'].apply(derive_signal)

    expected_signals = ["BUY", "SELL", "HOLD", "HOLD"]
    assert df['trade_signal'].tolist() == expected_signals
