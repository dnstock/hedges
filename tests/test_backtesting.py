import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.rl_trading.evaluate import run_backtest

@patch('src.rl_trading.evaluate.YahooDownloader')
@patch('src.rl_trading.evaluate.FeatureEngineer')
@patch('src.rl_trading.evaluate.StockTradingEnv')
@patch('src.rl_trading.evaluate.DRLAgent')
def test_backtest_execution(mock_drl_agent, mock_env, mock_fe, mock_downloader,
                          backtest_config, sample_market_data, sample_portfolio_history):
    """Test the full backtesting execution flow"""
    # Setup mocks
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_market_data
    mock_downloader.return_value = mock_downloader_instance

    mock_fe_instance = Mock()
    mock_fe_instance.preprocess_data.return_value = sample_market_data
    mock_fe.return_value = mock_fe_instance

    mock_env_instance = Mock()
    mock_env_instance.reset.return_value = np.zeros(10)  # Dummy observation
    mock_env_instance.step.side_effect = [
        (np.zeros(10), 0.1, False, {}),
        (np.zeros(10), 0.2, True, {})
    ]
    mock_env_instance.save_asset_memory.return_value = sample_portfolio_history
    mock_env.return_value = mock_env_instance

    mock_model = Mock()
    mock_model.predict.return_value = (np.zeros(2), None)  # (action, states)
    mock_agent_instance = Mock()
    mock_agent_instance.load_model.return_value = mock_model
    mock_drl_agent.return_value = mock_agent_instance

    # Run backtest
    with patch('pandas.DataFrame.to_csv') as mock_to_csv:
        run_backtest(backtest_config)

        # Verify the workflow
        mock_downloader.assert_called_once()
        mock_fe.assert_called_once()
        mock_env.assert_called_once()
        mock_drl_agent.assert_called_once()
        mock_to_csv.assert_called_once()

def test_performance_metrics(sample_portfolio_history):
    """Test calculation of performance metrics"""
    daily_returns = sample_portfolio_history['daily_return']

    # Test Sharpe Ratio calculation
    sharpe = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
    assert isinstance(sharpe, float)

    # Test total return calculation
    total_return = (sample_portfolio_history['total_asset'].iloc[-1] /
                   sample_portfolio_history['total_asset'].iloc[0]) - 1
    assert isinstance(total_return, float)

@patch('src.rl_trading.evaluate.YahooDownloader')
def test_data_preprocessing(mock_downloader, sample_market_data, backtest_config):
    """Test the data preprocessing steps"""
    mock_downloader_instance = Mock()
    mock_downloader_instance.fetch_data.return_value = sample_market_data
    mock_downloader.return_value = mock_downloader_instance

    with patch('src.rl_trading.evaluate.FeatureEngineer') as mock_fe:
        mock_fe_instance = Mock()
        mock_fe_instance.preprocess_data.return_value = sample_market_data
        mock_fe.return_value = mock_fe_instance

        run_backtest(backtest_config)

        # Verify preprocessing steps
        mock_downloader.assert_called_once()
        mock_fe.assert_called_once()
        mock_fe_instance.preprocess_data.assert_called_once()

def test_error_handling():
    """Test error handling for various failure scenarios"""
    invalid_config = {
        "env": {
            "ticker_list": [],  # Empty ticker list
            "start_date": "2023-01-01",
            "end_date": "2022-12-31"  # End date before start date
        }
    }

    with pytest.raises(ValueError):
        run_backtest(invalid_config)

    missing_model_config = {
        "env": {
            "ticker_list": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31"
        }
        # Missing model_filename
    }

    with pytest.raises(KeyError):
        run_backtest(missing_model_config)

def test_model_predictions(mock_model_output, sample_market_data, backtest_config):
    """Test model prediction handling"""
    with patch('src.rl_trading.evaluate.DRLAgent') as mock_agent:
        mock_model = Mock()
        mock_model.predict.return_value = (mock_model_output['actions'], None)

        mock_agent_instance = Mock()
        mock_agent_instance.load_model.return_value = mock_model
        mock_agent.return_value = mock_agent_instance

        with patch('src.rl_trading.evaluate.YahooDownloader') as mock_downloader:
            mock_downloader_instance = Mock()
            mock_downloader_instance.fetch_data.return_value = sample_market_data
            mock_downloader.return_value = mock_downloader_instance

            run_backtest(backtest_config)

            # Verify model was loaded and used
            mock_agent_instance.load_model.assert_called_once()
            assert mock_model.predict.called
