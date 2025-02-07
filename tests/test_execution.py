import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.execution.portfolio import Portfolio
from src.execution.live_trader import (
    run_live_trading,
    create_dummy_environment,
    fetch_latest_data,
    fetch_current_prices,
    prepare_observation,
    place_orders
)

class TestPortfolio:
    def test_portfolio_initialization(self):
        """Test portfolio initialization with different cash amounts"""
        portfolio = Portfolio(initial_cash=1000000)
        assert portfolio.cash == 1000000
        assert len(portfolio.positions) == 0

    def test_update_position(self):
        """Test position updates for buys and sells"""
        portfolio = Portfolio(initial_cash=1000000)

        # Test buying
        portfolio.update_position("AAPL", 100, 150.0)
        assert portfolio.positions["AAPL"] == 100
        assert portfolio.cash == 1000000 - (100 * 150.0)

        # Test selling
        portfolio.update_position("AAPL", -50, 160.0)
        assert portfolio.positions["AAPL"] == 50
        assert portfolio.cash == 1000000 - (100 * 150.0) + (50 * 160.0)

    def test_portfolio_valuation(self, mock_market_prices):
        """Test portfolio valuation with current market prices"""
        portfolio = Portfolio(initial_cash=1000000)

        # Add some positions
        portfolio.update_position("AAPL", 100, 145.0)
        portfolio.update_position("MSFT", 50, 290.0)

        # Calculate expected value
        expected_value = (
            portfolio.cash +
            100 * mock_market_prices["AAPL"] +
            50 * mock_market_prices["MSFT"]
        )

        assert portfolio.get_portfolio_value(mock_market_prices) == expected_value

    def test_position_removal(self):
        """Test that positions are removed when they reach 0"""
        portfolio = Portfolio(initial_cash=1000000)

        # Buy then fully sell
        portfolio.update_position("AAPL", 100, 150.0)
        portfolio.update_position("AAPL", -100, 160.0)

        assert "AAPL" not in portfolio.positions

    def test_portfolio_reset(self):
        """Test portfolio reset functionality"""
        portfolio = Portfolio(initial_cash=1000000)
        portfolio.update_position("AAPL", 100, 150.0)

        portfolio.reset()
        assert portfolio.cash == 1000000
        assert len(portfolio.positions) == 0


class TestLiveTrader:
    @patch('src.execution.live_trader.REST')
    @patch('src.execution.live_trader.DRLAgent')
    def test_live_trading_initialization(self, mock_drl_agent, mock_rest, execution_config):
        """Test live trading system initialization"""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        mock_model = Mock()
        mock_drl_agent.return_value.load_model.return_value = mock_model

        with patch('src.execution.live_trader.time.sleep') as mock_sleep:
            # Force exit after one iteration
            mock_sleep.side_effect = KeyboardInterrupt

            try:
                run_live_trading(execution_config)
            except KeyboardInterrupt:
                pass

            # Verify initialization
            mock_rest.assert_called_once()
            mock_drl_agent.return_value.load_model.assert_called_once()

    @patch('src.execution.live_trader.REST')
    def test_fetch_latest_data(self, mock_rest, mock_alpaca_bar):
        """Test market data fetching"""
        mock_api = Mock()
        mock_api.get_bars.return_value = [mock_alpaca_bar]
        mock_rest.return_value = mock_api

        tickers = ["AAPL", "MSFT"]
        df = fetch_latest_data(mock_api, tickers)

        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"date", "tic", "open", "high", "low", "close", "volume"}

    @patch('src.execution.live_trader.REST')
    def test_fetch_current_prices(self, mock_rest, mock_alpaca_quote):
        """Test current price fetching"""
        mock_api = Mock()
        mock_api.get_latest_quote.return_value = mock_alpaca_quote
        mock_rest.return_value = mock_api

        tickers = ["AAPL", "MSFT"]
        prices = fetch_current_prices(mock_api, tickers)

        assert isinstance(prices, dict)
        assert all(ticker in prices for ticker in tickers)

    def test_dummy_environment_creation(self, execution_config):
        """Test creation of dummy environment for model loading"""
        env = create_dummy_environment(execution_config)
        assert env is not None

        # Test reset provides correct observation shape
        obs = env.reset()
        expected_shape = len(execution_config["ticker_list"]) * 5  # Basic features
        assert len(obs) == expected_shape

    @patch('src.execution.live_trader.REST')
    def test_order_placement(self, mock_rest, mock_market_prices):
        """Test order placement logic"""
        mock_api = Mock()
        mock_rest.return_value = mock_api

        portfolio = Portfolio(initial_cash=1000000)
        ticker_list = ["AAPL", "MSFT"]
        action = [50, -30]  # Buy AAPL, Sell MSFT
        max_position_size = 100

        place_orders(
            mock_api, portfolio, ticker_list, action,
            max_position_size, mock_market_prices
        )

        # Verify order submissions
        assert mock_api.submit_order.call_count == 2

        # Verify portfolio updates
        assert portfolio.positions.get("AAPL") == 50
        assert portfolio.positions.get("MSFT") == -30

    def test_risk_management(self, execution_config):
        """Test risk management constraints"""
        portfolio = Portfolio(initial_cash=execution_config["initial_cash"])
        max_position_size = execution_config["risk_management"]["max_position_size"]

        # Test position size limit
        action = [150]  # Exceeds max_position_size
        ticker_list = ["AAPL"]
        current_prices = {"AAPL": 150.0}

        with patch('src.execution.live_trader.REST') as mock_rest:
            mock_api = Mock()
            mock_rest.return_value = mock_api

            place_orders(
                mock_api, portfolio, ticker_list, action,
                max_position_size, current_prices
            )

            # Verify order was capped at max_position_size
            order_args = mock_api.submit_order.call_args[1]
            assert int(order_args["qty"]) <= max_position_size
