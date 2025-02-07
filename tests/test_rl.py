import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.rl_trading.train import train_rl_agent

@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    return pd.DataFrame({
        'date': dates,
        'tic': ['AAPL'] * len(dates),
        'open': np.random.uniform(150, 160, len(dates)),
        'high': np.random.uniform(155, 165, len(dates)),
        'low': np.random.uniform(145, 155, len(dates)),
        'close': np.random.uniform(150, 160, len(dates)),
        'volume': np.random.randint(1000000, 2000000, len(dates))
    })

@pytest.fixture
def rl_config():
    """Sample RL configuration for testing"""
    return {
        "env": {
            "ticker_list": ["AAPL", "MSFT"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "time_interval": "1D"
        },
        "training": {
            "timesteps": 1000,
            "learning_rate": 0.001,
            "batch_size": 64
        },
        "agent": "PPO"
    }

@patch('src.rl_trading.train.YahooDownloader')
@patch('src.rl_trading.train.FeatureEngineer')
@patch('src.rl_trading.train.StockTradingEnv')
@patch('src.rl_trading.train.DRLAgent')
def test_train_rl_agent(mock_drl_agent, mock_env, mock_fe, mock_downloader,
                       rl_config, sample_stock_data):
    # Setup mocks
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_stock_data
    mock_downloader.return_value = mock_downloader_instance

    mock_fe_instance = Mock()
    mock_fe_instance.preprocess_data.return_value = sample_stock_data
    mock_fe.return_value = mock_fe_instance

    mock_env_instance = Mock()
    mock_env.return_value = mock_env_instance

    mock_agent_instance = Mock()
    mock_model = Mock()
    mock_agent_instance.get_model.return_value = mock_model
    mock_agent_instance.train_model.return_value = mock_model
    mock_drl_agent.return_value = mock_agent_instance

    # Test training
    train_rl_agent(rl_config)

    # Verify interactions
    mock_downloader.assert_called_once()
    mock_fe.assert_called_once()
    mock_env.assert_called_once()
    mock_drl_agent.assert_called_once()
    mock_agent_instance.get_model.assert_called_once()
    mock_agent_instance.train_model.assert_called_once()
    mock_model.save.assert_called_once()

def test_rl_config_validation():
    """Test validation of RL configuration"""
    invalid_config = {
        "env": {
            "ticker_list": [],  # Empty ticker list
            "start_date": "2023-01-01",
            "end_date": "2022-12-31"  # End date before start date
        },
        "training": {
            "timesteps": -1000  # Invalid timesteps
        }
    }

    with pytest.raises(ValueError):
        train_rl_agent(invalid_config)

@patch('src.rl_trading.train.YahooDownloader')
def test_data_preprocessing(mock_downloader, sample_stock_data, rl_config):
    """Test data preprocessing steps"""
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_stock_data
    mock_downloader.return_value = mock_downloader_instance

    with patch('src.rl_trading.train.FeatureEngineer') as mock_fe:
        mock_fe_instance = Mock()
        mock_fe_instance.preprocess_data.return_value = sample_stock_data
        mock_fe.return_value = mock_fe_instance

        train_rl_agent(rl_config)

        # Verify preprocessing steps
        mock_fe.assert_called_once()
        mock_fe_instance.preprocess_data.assert_called_once()

@patch('src.rl_trading.train.YahooDownloader')
@patch('src.rl_trading.train.FeatureEngineer')
@patch('src.rl_trading.train.StockTradingEnv')
@patch('src.rl_trading.train.DRLAgent')
def test_different_agents(mock_drl_agent, mock_env, mock_fe, mock_downloader,
                        rl_config, sample_stock_data):
    """Test different RL agents (PPO, A2C)"""
    # Setup basic mocks
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_stock_data
    mock_downloader.return_value = mock_downloader_instance

    mock_fe_instance = Mock()
    mock_fe_instance.preprocess_data.return_value = sample_stock_data
    mock_fe.return_value = mock_fe_instance

    mock_env_instance = Mock()
    mock_env.return_value = mock_env_instance

    # Test PPO
    rl_config["agent"] = "PPO"
    train_rl_agent(rl_config)
    assert mock_drl_agent().get_model.call_args[0][0] == "ppo"

    # Test A2C
    rl_config["agent"] = "A2C"
    train_rl_agent(rl_config)
    assert mock_drl_agent().get_model.call_args[0][0] == "a2c"

    # Test invalid agent
    rl_config["agent"] = "INVALID"
    with pytest.raises(ValueError):
        train_rl_agent(rl_config)
