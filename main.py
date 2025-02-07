"""
Main entry point for the Autonomous Hedge Fund project.

Usage:
    python main.py --mode nlp
    python main.py --mode rl
    python main.py --mode live
    python main.py --mode backtest
    python main.py --mode all (default - runs all modules except live trading)
"""
import logging
import argparse
from src.nlp_signals.pipeline import run_nlp_pipeline
from src.rl_trading.train import train_rl_agent
from src.rl_trading.evaluate import run_backtest
from src.execution.live_trader import run_live_trading
from src.utils.config_loader import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Hedges: Autonomous Investment Fund")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        help="Run mode: 'nlp', 'rl', 'live', 'backtest', or 'all'"
    )
    return parser.parse_args()

def main():
    # Setup logging (optional)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[
                            logging.StreamHandler(),  # Log to stdout
                            logging.FileHandler("logs/hedges.log"),
                        ])

    # Read command-line arguments
    args = parse_args()

    # Load configuration settings
    config = load_config()

    # Decide which modules to run based on the --mode argument
    if args.mode in ("nlp", "all"):
        logging.info("Running NLP pipeline...")
        run_nlp_pipeline(config["nlp_signals"])

    if args.mode in ("rl", "all"):
        logging.info("Training RL agent...")
        train_rl_agent(config["rl_trading"])

    if args.mode in ("backtest", "all"):
        logging.info("Running backtest...")
        run_backtest(config["rl_trading"])

    if args.mode in ("live"):
        if config["execution"].get("enabled", False):
            logging.info("Starting live trader...")
            run_live_trading(config["execution"])
        else:
            logging.warning("Live trading is disabled in settings.yaml")

if __name__ == "__main__":
    main()
