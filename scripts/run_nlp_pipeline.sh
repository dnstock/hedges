#!/usr/bin/env bash
set -e  # Exit on error

# 1. (Optional) Activate a Python virtual environment:
#    Adjust this path if you use a different directory name.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 2. Run only the NLP pipeline
python main.py --mode nlp
