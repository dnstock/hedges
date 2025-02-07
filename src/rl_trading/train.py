import logging
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

def train_rl_agent(rl_config):
    """
    Train an RL agent for trading using FinRL.
    Expects 'rl_config' to contain 'env' and 'training' sections with:
      - env.ticker_list (list of stock tickers)
      - env.start_date, env.end_date
      - env.time_interval (e.g. "1D")
      - training.timesteps
      - training.learning_rate
      - agent (e.g. "PPO")
    """

    logging.info("Starting RL training with config: %s", rl_config)

    # 1. Parse config
    ticker_list = rl_config["env"]["ticker_list"]
    start_date = rl_config["env"]["start_date"]      # "2020-01-01"
    end_date = rl_config["env"]["end_date"]          # "2021-01-01"
    time_interval = rl_config["env"]["time_interval"] # e.g. "1D"
    agent_name = rl_config.get("agent", "PPO")        # default to PPO
    timesteps = rl_config["training"]["timesteps"]
    learning_rate = rl_config["training"]["learning_rate"]
    batch_size = rl_config["training"].get("batch_size", 64)

    # 2. Download historical data (Yahoo is a built-in option in FinRL)
    downloader = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
        time_interval=time_interval,
    )
    df_raw = downloader.fetch_data()

    # Resample data to a lower frequency (e.g. weekly)
    df_raw = df_raw.set_index("date").resample("W").agg({...})

    # 3. Feature engineering (technical indicators, turbulence)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
        use_turbulence=True,
        user_defined_feature=False
    )
    df_processed = fe.preprocess_data(df_raw)

    # 4. Create trading environment
    #    For a quick start, pass the entire dataset for "training"
    #    but in practice, split into train/test or train/validation
    env = StockTradingEnv(
        df=df_processed,
        stock_dim=len(ticker_list),
        hmax=100,                # max shares per trade
        initial_amount=1e6,      # starting capital
        transaction_cost_pct=0.001,
        reward_scaling=1e-4,
        state_space=len(ticker_list)*5,  # simplistic guess
        action_space=len(ticker_list),
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"]
        # You can add other custom params here
    )

    # 5. Initialize DRL Agent (PPO, A2C, DDPG, etc.)
    agent = DRLAgent(env=env)
    agent_name_upper = agent_name.upper()
    if agent_name_upper == "PPO":
        model = agent.get_model(
            "ppo",
            model_kwargs={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )
    elif agent_name_upper == "A2C":
        model = agent.get_model(
            "a2c",
            model_kwargs={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # 6. Train the RL model
    trained_model = agent.train_model(
        model=model,
        tb_log_name=agent_name_upper,
        total_timesteps=timesteps
    )

    # 7. Save the trained model for future reuse
    model_filename = f"{agent_name_upper}_model.zip"
    trained_model.save(model_filename)
    logging.info("RL training complete. Model saved as: %s", model_filename)
