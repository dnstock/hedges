# Hedges | AI-Powered Trading Platform

Hedges is an autonomous investment engine that uses NLP and Reinforcement Learning to generate trading signals and execute trades.  

It is designed to be modular and extensible, allowing for easy experimentation with different models and strategies.

## Project Structure
```plaintext
hedges/
├── README.md
├── pyproject.toml          # manage Python dependencies
├── config/
│   └── settings.yaml       # central config for API keys, model params, etc.
├── data/
│   ├── raw/                # unprocessed data dumps (news, market data, etc.)
│   ├── interim/            # any partial cleaning or intermediate files
│   └── processed/          # final, cleaned data ready for modeling
├── src/
│   ├── nlp_signals/        # FinGPT (or other NLP) logic
│   │   ├── __init__.py
│   │   ├── pipeline.py     # text ingestion, cleaning, feature extraction
│   │   └── sentiment.py    # e.g. sentiment classification
│   ├── rl_trading/         # FinRL logic
│   │   ├── __init__.py
│   │   ├── train.py        # RL agent training
│   │   └── evaluate.py     # backtesting & performance metrics
│   └── execution/
│       ├── __init__.py
│       ├── live_trader.py  # broker connections, order execution
│       └── portfolio.py    # basic portfolio tracking
├── tests/
|   ├── conftest.py         # pytest fixtures
│   ├── test_nlp.py         # test NLP pipeline
│   │── test_rl.py          # test RL training
│   ├── test_backtesting.py # test RL backtesting
│   └── test_execution.py   # test live trading
└── scripts/
    ├── run_nlp_pipeline.sh
    ├── run_rl_training.sh
    ├── run_backtest.sh
    ├── run_live_trading.sh
    ├── run_tests.sh
    └── run_all.sh
```

## Installation
This project uses [Poetry](https://python-poetry.org/) for dependency management. Install it first if you don't have it already.
```bash
pip install poetry
```

Clone the repository and install dependencies:
```bash
git clone git@github.com:dnstock/hedges.git
cd hedges
poetry install
```

## Configuration

### Create Settings File
API keys, model hyperparameters, and other settings are stored in `config/settings.yaml`.  

Create it by copying `config/settings_template.yaml` and modifying in the necessary values.
```bash
cp config/settings_template.yaml config/settings.yaml
```

### Create Environment File (optional)
Sensitive information like API keys can optionally be stored in an `.env` file.  

Any variables defined here will override those in `config/settings.yaml`.
```bash
cp .env_template .env
```

## Usage
From the project root directory:
```bash
./scripts/run_nlp_pipeline.sh   # run NLP pipeline
./scripts/run_rl_training.sh    # train RL agent
./scripts/run_backtest.sh       # evaluate agent performance
./scripts/run_live_trading.sh   # execute or simulate live trades
./scripts/run_all.sh            # runs all modules (except live trading)
```

## Testing
Run tests with pytest via the `run_tests.sh` script:
```bash
./scripts/run_tests.sh          # run all tests (default)
./scripts/run_tests.sh nlp      # run NLP pipeline tests
./scripts/run_tests.sh rl       # run RL training tests
./scripts/run_tests.sh bt       # run backtesting tests
./scripts/run_tests.sh lt       # run live trading tests
```

## Contributing

Contributions are welcome! 

To contribute, please follow these steps:

1. Fork the repository

```sh
gh repo fork git@github.com:dnstock/hedges.git
```

2. Create a new branch

```sh
git checkout -b feature/my-feature
```

3. Make your changes

```sh
git add .
git commit -m "Add my feature"
```

4. Commit your changes to your fork

```sh
git push origin feature/my-feature
```

5. Create a pull request

```sh
gh pr create
```

6. Wait for review and approval

```sh
gh pr status
```

7. Make any requested changes

```sh
git add .
git commit -m "Address review comments"
git push origin feature/my-feature
```

8. Merge your changes

```sh
gh pr merge
```

9. Celebrate your contribution!

```sh
🎉
```

## Support

For support, please contact the author.

## Authors

- [Dan Harcsztark (dnstock)](https://github.com/dnstock)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
