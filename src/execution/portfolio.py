import logging

class Portfolio:
    """
    A simple in-memory portfolio tracker.
    Assumes a single currency (e.g. USD).
    """

    def __init__(self, initial_cash=1e6):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # positions[ticker] = number_of_shares
        self.positions = {}

    def update_position(self, ticker: str, shares_delta: int, fill_price: float):
        """
        Adjust position for a single ticker and update cash.
          - shares_delta > 0 => bought shares
          - shares_delta < 0 => sold shares
          - fill_price => the average fill price for those shares
        """
        if ticker not in self.positions:
            self.positions[ticker] = 0

        cost = fill_price * abs(shares_delta)
        # If shares_delta > 0, we spend cost
        # If shares_delta < 0, we gain cost
        if shares_delta > 0:
            self.cash -= cost
        else:
            self.cash += cost

        # Update the position
        self.positions[ticker] += shares_delta

        # If position goes to 0, optionally remove from dict
        if self.positions[ticker] == 0:
            del self.positions[ticker]

        logging.info(
            f"Updated position for {ticker}, delta={shares_delta}, fill_price={fill_price:.2f}."
            f" New shares={self.positions.get(ticker, 0)}, Remaining cash={self.cash:.2f}"
        )

    def get_portfolio_value(self, market_prices: dict) -> float:
        """
        Returns total portfolio value = cash + sum(position_shares * current_price).
        'market_prices' is expected to be a dict like { 'AAPL': 175.0, 'MSFT': 325.0, ... }
        """
        value = self.cash
        for ticker, shares in self.positions.items():
            price = market_prices.get(ticker, 0.0)
            value += shares * price
        return value

    def get_positions(self):
        """
        Returns a copy of the positions dict for inspection or logging.
        """
        return dict(self.positions)

    def reset(self):
        """Resets the portfolio to the initial cash and empties positions."""
        self.cash = self.initial_cash
        self.positions.clear()
        logging.info("Portfolio reset to initial cash.")
