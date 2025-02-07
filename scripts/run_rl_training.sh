#!/usr/bin/env bash
set -e

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 2. Run only the RL training
python main.py --mode rl
