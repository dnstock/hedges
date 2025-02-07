import logging
import time
import pandas as pd
from alpaca_trade_api import REST, TimeFrame

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

from src.execution.portfolio import Portfolio

def run_live_trading(execution_config):
    """
    Continuously fetches real-time data from Alpaca, runs inference with a trained RL model,
    and places orders subject to basic risk management rules.
    """

    # 1. Parse Alpaca broker config
    api_key = execution_config["api_key"]
    api_secret = execution_config["api_secret"]
    paper_trading = execution_config.get("paper_trading", True)
    base_url = "https://paper-api.alpaca.markets" if paper_trading else "https://api.alpaca.markets"
    base_currency = execution_config.get("base_currency", "USD")
    initial_cash = execution_config.get("initial_cash", 1e6)

    logging.info("Connecting to Alpaca. Paper trading: %s", paper_trading)
    logging.info("Base currency: %s", base_currency)

    alpaca_api = REST(api_key, api_secret, base_url, api_version="v2")

    # 2. Risk management config
    risk_config = execution_config.get("risk_management", {})
    max_position_size = risk_config.get("max_position_size", 100)  # e.g., 100 shares
    daily_loss_limit = risk_config.get("daily_loss_limit", 500.0)  # e.g., 500 USD

    logging.info("Risk config => max_position_size: %d, daily_loss_limit: %.2f",
                 max_position_size, daily_loss_limit)

    # 3. Create or load a local portfolio tracker (for daily P&L checks)
    portfolio = Portfolio(initial_cash=initial_cash)
    daily_start_value = portfolio.get_portfolio_value({})  # no prices yet => just cash

    # 4. Load your RL model
    model_path = execution_config.get("model_filename", "PPO_model.zip")
    logging.info("Loading RL model from: %s", model_path)
    dummy_env = create_dummy_environment(execution_config)
    agent = DRLAgent(env=dummy_env)
    model = agent.load_model(model_path)

    # 5. Live trading loop
    loop_interval = execution_config.get("loop_interval_sec", 60)
    ticker_list = execution_config.get("ticker_list", ["AAPL", "MSFT"])

    while True:
        try:
            # a) Check daily loss limit before trading
            current_market_prices = fetch_current_prices(alpaca_api, ticker_list)
            current_portfolio_value = portfolio.get_portfolio_value(current_market_prices)
            current_drawdown = daily_start_value - current_portfolio_value
            if current_drawdown >= daily_loss_limit:
                logging.warning(
                    "Daily loss limit reached (%.2f >= %.2f). Halting trading for the day.",
                    current_drawdown, daily_loss_limit
                )
                time.sleep(loop_interval)
                continue

            # b) Fetch latest market data for observation
            df_market = fetch_latest_data(alpaca_api, ticker_list)
            obs = prepare_observation(dummy_env, df_market)

            # c) Get action from RL model
            action, _states = model.predict(obs, deterministic=True)

            # d) Place orders subject to max_position_size
            place_orders(
                api=alpaca_api,
                portfolio=portfolio,
                ticker_list=ticker_list,
                action=action,
                max_position_size=max_position_size,
                current_prices=current_market_prices
            )

            # e) Sleep until next cycle
            time.sleep(loop_interval)

        except Exception as e:
            logging.error("Error in live trading loop: %s", e)
            time.sleep(30)  # brief pause before retry


def create_dummy_environment(execution_config):
    """
    Creates a minimal StockTradingEnv to match observation/action space
    of the trained model. Used mainly so we can load the model in memory.
    """
    ticker_list = execution_config.get("ticker_list", ["AAPL", "MSFT"])
    dummy_df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=10),
        "tic": [ticker_list[0]] * 10,
        "close": [100.0]*10,
        "high": [101.0]*10,
        "low": [99.0]*10,
        "open": [100.5]*10,
        "volume": [1000]*10
    })
    for tic in ticker_list[1:]:
        tmp = dummy_df.copy()
        tmp["tic"] = tic
        dummy_df = pd.concat([dummy_df, tmp], ignore_index=True)

    env = StockTradingEnv(
        df=dummy_df,
        stock_dim=len(ticker_list),
        hmax=100,
        initial_amount=1e6,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=len(ticker_list)*5,
        action_space=len(ticker_list),
        tech_indicator_list=[],
    )
    return env


def fetch_latest_data(api: REST, tickers):
    """
    Grabs the latest bar data for each ticker from Alpaca.
    Returns a DataFrame with columns expected by the environment.
    """
    end_time = pd.Timestamp.now(tz="UTC")
    start_time = end_time - pd.Timedelta(minutes=1)

    all_data = []
    for ticker in tickers:
        bars = api.get_bars(
            ticker,
            TimeFrame.Minute,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            limit=1
        )
        if bars:
            bar = bars[0]
            all_data.append({
                "date": bar.t,
                "tic": ticker,
                "open": bar.o,
                "high": bar.h,
                "low": bar.l,
                "close": bar.c,
                "volume": bar.v
            })
        else:
            logging.warning("No data returned for %s in the last minute.", ticker)
    df = pd.DataFrame(all_data)
    return df


def fetch_current_prices(api: REST, tickers):
    """
    Fetches the latest quote for each ticker to estimate current market prices
    for portfolio valuation. Returns a dict { 'AAPL': 175.32, 'MSFT': 321.11, ... }.
    """
    prices = {}
    for ticker in tickers:
        try:
            quote = api.get_latest_quote(ticker)
            prices[ticker] = quote.ap  # ask price (or mid, or last trade)
        except Exception as e:
            logging.error("Failed to get quote for %s: %s", ticker, e)
            prices[ticker] = 0.0
    return prices


def prepare_observation(env: StockTradingEnv, df_market: pd.DataFrame):
    """
    Converts real-time data into an observation for the environment.
    This minimal version simply resets the dummy env (since we're not
    truly stepping in real-time). A more advanced approach would
    incorporate 'df_market' to adjust env state.
    """
    obs = env.reset()
    return obs


def place_orders(api: REST,
                 portfolio,
                 ticker_list,
                 action,
                 max_position_size,
                 current_prices):
    """
    Converts RL model actions into broker orders while respecting max_position_size.
    In a more advanced system, you'd validate positions vs. portfolio as well.
    """
    logging.info("Placing orders for tickers: %s with action: %s", ticker_list, action)

    for ticker, amt in zip(ticker_list, action):
        shares = int(round(amt))
        # Risk check #1: limit position size
        if abs(shares) > max_position_size:
            logging.warning(
                "Requested shares (%d) exceed max_position_size (%d) for %s. Capping.",
                shares, max_position_size, ticker
            )
            shares = max_position_size if shares > 0 else -max_position_size

        if shares == 0:
            logging.debug("No action for %s (action=0).", ticker)
            continue

        # BUY
        if shares > 0:
            try:
                fill_price = current_prices.get(ticker, 0.0)
                logging.info(f"Submitting BUY order => {ticker}, shares={shares}, price≈{fill_price:.2f}")
                # Place order
                api.submit_order(
                    symbol=ticker,
                    qty=shares,
                    side="buy",
                    type="market",
                    time_in_force="gtc"
                )
                # Update local portfolio immediately (assuming full fill at 'fill_price')
                portfolio.update_position(ticker, shares, fill_price)

            except Exception as e:
                logging.error("Failed to place BUY order for %s: %s", ticker, e)

        # SELL
        elif shares < 0:
            shares_to_sell = abs(shares)
            try:
                fill_price = current_prices.get(ticker, 0.0)
                logging.info(f"Submitting SELL order => {ticker}, shares={shares_to_sell}, price≈{fill_price:.2f}")
                # Place order
                api.submit_order(
                    symbol=ticker,
                    qty=shares_to_sell,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                # Update local portfolio
                portfolio.update_position(ticker, -shares_to_sell, fill_price)

            except Exception as e:
                logging.error("Failed to place SELL order for %s: %s", ticker, e)
