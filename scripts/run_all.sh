#!/usr/bin/env bash
set -e  # Exit on any error

# 1. (Optional) Activate your Python virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 2. Run the NLP pipeline
echo "Running NLP pipeline..."
python main.py --mode nlp

# 3. Train the RL agent
echo "Training the RL agent..."
python main.py --mode rl

# 4. (Optional) Run backtesting
#    Only do this if you have defined a "backtest" mode in main.py
#    Otherwise, skip or reuse the RL mode if it already includes a backtest.
echo "Backtesting the strategy..."
python main.py --mode backtest

echo "All steps completed!"
