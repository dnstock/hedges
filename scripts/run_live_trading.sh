#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 2. Run the live trading mode
python main.py --mode live
