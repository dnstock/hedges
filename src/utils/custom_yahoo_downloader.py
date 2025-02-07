import time
import logging
import pandas as pd
import yfinance as yf
from requests.exceptions import ReadTimeout

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

class CustomYahooDownloader(YahooDownloader):
    def __init__(self, start_date, end_date, ticker_list, interval, timeout, proxy):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list
        self.interval = interval
        self.timeout = timeout
        self.proxy = proxy

    def _fetch(self, tic: str) -> pd.DataFrame:
        """
        Fetch data for a single ticker.
        """
        return yf.download(
            tic,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            timeout=self.timeout,
            proxy=self.proxy,
        )

    def _process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the raw data for FeatureEngineer
        """
        # reset the index, we want to use numbers as index instead of dates
        df = df.reset_index()
        # convert the column names to standardized names
        df.columns = [
            'date',
            'tic',
            'open',
            'high',
            'low',
            'close',
            'volume',
        ]
        # create day of the week column (monday = 0)
        df['day'] = df['date'].dt.dayofweek
        # convert date to standard string format, easy to filter
        df['date'] = pd.to_datetime(df['date'])
        # Drop missing data
        df = df.dropna()
        df = df.reset_index(drop=True)

        logging.info(f"Data fetched successfully. Shape: {df.shape}")

        return df.sort_values(by=["date", "tic"]).reset_index(drop=True)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch data using yfinance
        """
        logging.info(f"Fetching data for {self.ticker_list} from {self.start_date} to {self.end_date}")
        data_df = pd.DataFrame()
        num_failures = 0
        for tic in self.ticker_list:
            temp_df = self._fetch(tic)
            if temp_df.empty:
                num_failures += 1
            else:
                temp_df.insert(1, "tic", tic)
                data_df = pd.concat([data_df, temp_df], axis=0)

        if num_failures == len(self.ticker_list):
            raise ValueError("No data fetched.")

        return self._process(data_df)

    def fetch_data_with_retry(self, max_retries: int, delay: int) -> pd.DataFrame:
        """
        Attempt to fetch data multiple times before raising an error.
        """
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            for attempt in range(1, max_retries + 1):
                try:
                    temp_df = self._fetch(tic)
                    if not temp_df.empty:
                        temp_df.insert(1, "tic", tic)
                        data_df = pd.concat([data_df, temp_df], axis=0)
                        break
                    logging.warning("Empty DataFrame returned on attempt %d.", attempt)
                except ReadTimeout as e:
                    logging.warning("Attempt %d/%d failed due to timeout: %s", attempt, max_retries, e)
                except Exception as e:
                    logging.error("Attempt %d/%d encountered an error: %s", attempt, max_retries, e)
                time.sleep(delay)
            if data_df.empty:
                raise TimeoutError("Maximum retries reached. Failed to fetch data for ticker: %s", tic)

        return self._process(data_df)
