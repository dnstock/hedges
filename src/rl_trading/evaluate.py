import logging
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

def run_backtest(rl_config):
    """
    Loads a previously trained RL model and runs a backtest.
    Expects 'rl_config' with:
      - env.ticker_list, env.start_date, env.end_date
      - model filename (e.g. "PPO_model.zip")
    """

    logging.info("Starting RL backtest with config: %s", rl_config)

    ticker_list = rl_config["env"]["ticker_list"]
    start_date = rl_config["env"]["start_date"]
    end_date = rl_config["env"]["end_date"]
    agent_name = rl_config.get("agent", "PPO").upper()
    model_filename = rl_config.get("model_filename", f"{agent_name}_model.zip")

    # 1. Download & preprocess data (for backtest period)
    downloader = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list
    )
    df_raw = downloader.fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
        use_turbulence=True,
        user_defined_feature=False
    )
    df_processed = fe.preprocess_data(df_raw)

    # 2. Create environment in "test" mode
    env_test = StockTradingEnv(
        df=df_processed,
        stock_dim=len(ticker_list),
        hmax=100,
        initial_amount=1e6,
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=len(ticker_list)*5,
        action_space=len(ticker_list),
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"]
    )

    # 3. Load the trained model
    agent = DRLAgent(env=env_test)
    model = agent.load_model(model_filename)

    # 4. Run the model on the environment step-by-step to record performance
    obs = env_test.reset()
    done = False
    while not done:
        action = model.predict(obs)[0]  # stable-baselines3 returns (action, _states)
        obs, rewards, done, info = env_test.step(action)

    # 5. Evaluate results (P&L, Sharpe, etc.)
    df_total_value = env_test.save_asset_memory()
    daily_returns = df_total_value["total_asset"].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
    final_portfolio_value = df_total_value["total_asset"].iloc[-1]

    logging.info("Backtest complete. Final Portfolio Value: %.2f", final_portfolio_value)
    logging.info("Approximate Sharpe Ratio: %.2f", sharpe_ratio)

    # Optionally save a CSV for further analysis
    df_total_value.to_csv("backtest_results.csv", index=False)
    logging.info("Saved backtest results to backtest_results.csv.")
