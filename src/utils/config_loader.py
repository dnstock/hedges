import os
import yaml
from dotenv import load_dotenv

def load_config(yaml_path="config/settings.yaml"):
    """Loads settings from YAML, then overrides with environment variables if present."""
    # 1. Load environment variables from .env
    load_dotenv()  # by default, loads from '.env' in the current directory

    # 2. Load the base config from settings.yaml
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # 3. Override relevant fields with environment variables

    # - NLP signals
    #   If FINGPT_MODEL_NAME is set, override model_checkpoint
    if "nlp_signals" in config:
        config["nlp_signals"]["model_checkpoint"] = os.getenv(
            "FINGPT_MODEL_NAME",
            config["nlp_signals"].get("model_checkpoint")
        )
        config["nlp_signals"]["text_data_path"] = os.getenv(
            "FINGPT_DATA_DIR",
            config["nlp_signals"].get("text_data_path")
        )

    # - RL trading
    if "rl_trading" in config:
        # Example: override broker config if environment variables are present
        config["rl_trading"]["broker"] = os.getenv(
            "FINRL_BROKER",
            config["rl_trading"].get("broker", "alpaca")  # default "alpaca"
        )
        config["rl_trading"]["paper_trading"] = os.getenv(
            "FINRL_PAPER_TRADING",
            config["rl_trading"].get("paper_trading", True)
        )

    # - Execution
    if "execution" in config:
        config["execution"]["api_key"] = os.getenv(
            "FINRL_API_KEY",
            config["execution"].get("api_key", "")
        )
        config["execution"]["api_secret"] = os.getenv(
            "FINRL_API_SECRET",
            config["execution"].get("api_secret", "")
        )

    return config
