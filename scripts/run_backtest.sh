#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 2. Run a backtest (could reuse "rl" mode or create a custom mode)
python main.py --mode rl
