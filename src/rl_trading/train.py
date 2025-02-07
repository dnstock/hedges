import logging

from src.utils.custom_yahoo_downloader import CustomYahooDownloader as YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

def train_rl_agent(rl_config: dict):
    """
    Train an RL agent for trading using the provided configuration.
    """
    logging.info("Starting RL training with config: %s", rl_config)

    # --- Parse configuration ---
    agent_name = rl_config.get("agent", "PPO").upper()
    env_config = rl_config.get("env", {})
    download_config = rl_config.get("download", {})
    training_config = rl_config.get("training", {})

    ticker_list = env_config.get("ticker_list", [])
    start_date = env_config.get("start_date")
    end_date = env_config.get("end_date")
    time_interval = env_config.get("time_interval", "1D")
    if not ticker_list or not start_date or not end_date:
        raise ValueError("Missing required environment configuration (ticker_list, start_date, or end_date).")

    timeout = download_config.get("timeout", 30)
    max_retries = download_config.get("max_retries", 3)
    retry_delay = download_config.get("retry_delay", 5)
    proxy = download_config.get("proxy", None)

    timesteps = int(training_config.get("timesteps", 1e5))
    learning_rate = training_config.get("learning_rate", 0.00025)
    batch_size = training_config.get("batch_size", 64)
    initial_capital = training_config.get("initial_capital", 1e6)

    # --- Data Download ---
    downloader = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=ticker_list,
        interval=time_interval,
        timeout=timeout,
        proxy=proxy,
    )
    try:
        df_raw = downloader.fetch_data_with_retry(max_retries, retry_delay)
    except Exception as e:
        logging.error("Data fetching failed: %s", e)
        return
    # Resample data to a lower frequency (e.g. weekly)
    # df_raw = df_raw.set_index("date").resample("W").agg({...})

    # --- Feature Engineering ---
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=["macd", "rsi_30", "cci_30", "dx_30"],
        use_turbulence=True,
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df_raw)

    # --- Create Trading Environment ---
    #  For a quick start, pass the entire dataset for "training"
    #  but in practice, split into train/test or train/validation
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

    # --- Initialize DRL Agent ---
    agent = DRLAgent(env=env)
    if agent_name == "PPO":
        model = agent.get_model(
            "ppo",
            model_kwargs={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )
    elif agent_name == "A2C":
        model = agent.get_model(
            "a2c",
            model_kwargs={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
            }
        )
    elif agent_name == "DDPG":
        model = agent.get_model("ddpg")
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    # --- Train the RL Model ---
    logging.info("Training the model (%s)...", agent_name)
    trained_model = agent.train_model(
        model=model,
        tb_log_name=agent_name,
        total_timesteps=timesteps
    )
    logging.info("Training complete.")

    # --- Save the trained model ---
    model_filename = f"data/trained/{agent_name}_model.zip"
    trained_model.save(model_filename)
    logging.info("RL training complete. Model saved as: %s", model_filename)

    return trained_model
