import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.rl_trading.train import train_rl_agent

# Remove sample_stock_data fixture since we're using sample_market_data from conftest.py
# Remove rl_config fixture since we're using rl_training_config from conftest.py

@patch('src.rl_trading.train.YahooDownloader')
@patch('src.rl_trading.train.FeatureEngineer')
@patch('src.rl_trading.train.StockTradingEnv')
@patch('src.rl_trading.train.DRLAgent')
def test_train_rl_agent(mock_drl_agent, mock_env, mock_fe, mock_downloader,
                       rl_training_config, sample_market_data, env_config):
    """Test the full training pipeline with technical indicators"""
    # Setup mocks
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_market_data
    mock_downloader.return_value = mock_downloader_instance

    mock_fe_instance = Mock()
    mock_fe_instance.preprocess_data.return_value = sample_market_data
    mock_fe.return_value = mock_fe_instance

    mock_env_instance = Mock()
    mock_env.return_value = mock_env_instance

    mock_agent_instance = Mock()
    mock_model = Mock()
    mock_agent_instance.get_model.return_value = mock_model
    mock_agent_instance.train_model.return_value = mock_model
    mock_drl_agent.return_value = mock_agent_instance

    # Test training
    train_rl_agent(rl_training_config)

    # Verify the environment was created with correct technical indicators
    _, kwargs = mock_env.call_args
    assert kwargs['tech_indicator_list'] == env_config['tech_indicator_list']
    assert all(indicator in sample_market_data.columns
              for indicator in env_config['tech_indicator_list'])

def test_technical_indicators(sample_market_data):
    """Test the presence and validity of technical indicators"""
    required_indicators = ['macd', 'rsi_30', 'cci_30', 'dx_30']

    # Check presence of indicators
    for indicator in required_indicators:
        assert indicator in sample_market_data.columns

    # Verify indicator values are within expected ranges
    assert all(sample_market_data['rsi_30'].between(0, 100))
    assert all(sample_market_data['cci_30'].between(-200, 200))
    assert all(sample_market_data['dx_30'].between(0, 100))

@patch('src.rl_trading.train.YahooDownloader')
def test_data_preprocessing(mock_downloader, sample_market_data, rl_training_config, env_config):
    """Test data preprocessing with technical indicators"""
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_market_data
    mock_downloader.return_value = mock_downloader_instance

    with patch('src.rl_trading.train.FeatureEngineer') as mock_fe:
        mock_fe_instance = Mock()
        processed_data = sample_market_data.copy()
        mock_fe_instance.preprocess_data.return_value = processed_data
        mock_fe.return_value = mock_fe_instance

        train_rl_agent(rl_training_config)

        # Verify FeatureEngineer was called with correct indicator list
        _, kwargs = mock_fe.call_args
        assert kwargs.get('tech_indicator_list', []) == env_config['tech_indicator_list']

@patch('src.rl_trading.train.YahooDownloader')
@patch('src.rl_trading.train.FeatureEngineer')
@patch('src.rl_trading.train.StockTradingEnv')
@patch('src.rl_trading.train.DRLAgent')
def test_different_agents(mock_drl_agent, mock_env, mock_fe, mock_downloader,
                        rl_training_config, sample_market_data):
    """Test different RL agents with technical indicators"""
    # Setup basic mocks
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_market_data
    mock_downloader.return_value = mock_downloader_instance

    mock_fe_instance = Mock()
    mock_fe_instance.preprocess_data.return_value = sample_market_data
    mock_fe.return_value = mock_fe_instance

    mock_env_instance = Mock()
    mock_env.return_value = mock_env_instance

    # Test PPO configuration
    rl_training_config["agent"] = "PPO"
    train_rl_agent(rl_training_config)
    ppo_call = mock_drl_agent().get_model.call_args
    assert ppo_call[0][0] == "ppo"
    assert "learning_rate" in ppo_call[1]["model_kwargs"]

    # Test A2C configuration
    rl_training_config["agent"] = "A2C"
    train_rl_agent(rl_training_config)
    a2c_call = mock_drl_agent().get_model.call_args
    assert a2c_call[0][0] == "a2c"
    assert "learning_rate" in a2c_call[1]["model_kwargs"]

def test_state_space_calculation(env_config, sample_market_data):
    """Test the state space calculation with technical indicators"""
    n_stocks = len(sample_market_data['tic'].unique())
    n_technical_indicators = len(env_config['tech_indicator_list'])

    # Basic market data features (OHLCV) + technical indicators per stock
    expected_state_space = n_stocks * (5 + n_technical_indicators)

    assert env_config['state_space'] == expected_state_space or \
           env_config['state_space'] == n_stocks * 5  # Allow for basic feature set too

def test_error_handling():
    """Test error handling with invalid configurations"""
    invalid_config = {
        "env": {
            "ticker_list": [],  # Empty ticker list
            "start_date": "2023-01-01",
            "end_date": "2022-12-31"  # End date before start date
        },
        "training": {
            "timesteps": -1000,  # Invalid timesteps
            "learning_rate": -0.001  # Invalid learning rate
        }
    }

    with pytest.raises(ValueError):
        train_rl_agent(invalid_config)
