#!/usr/bin/env bash
set -e

TEST_OPTION=$1

# Activate the virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

case $TEST_OPTION in
  nlp)
    echo "Running NLP pipeline tests"
    pytest tests/test_nlp.py -v
    ;;
  rl)
    echo "Running RL training tests"
    pytest tests/test_rl.py -v
    ;;
  bt|backtesting)
    echo "Running backtesting tests"
    pytest tests/test_backtesting.py -v
    ;;
  lt|livetrading)
    echo "Running live trading tests"
    pytest tests/test_execution.py -v
    ;;
  all|*)
    if [ -z $TEST_OPTION ]; then
      echo "Running all tests"
      pytest tests -v
    else
      echo "Invalid option: $TEST_OPTION"
      echo "Usage: $0 [nlp|rl|bt|lt|all]"
      exit 1
    fi
    ;;
esac
