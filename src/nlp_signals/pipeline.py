import logging
import pandas as pd

from src.nlp_signals.sentiment import (
    load_sentiment_pipeline,
    compute_sentiment_scores,
    map_label_to_value,
)

def run_nlp_pipeline(nlp_config):
    """
    Orchestrates NLP signal extraction using a sentiment analysis model.
    Reads text data, applies a pretrained model, and outputs final signals.
    """

    # 1. Load data
    text_data_path = nlp_config.get("text_data_path", "data/processed/news_data.csv")
    logging.info("Loading text data from: %s", text_data_path)
    df = pd.read_csv(text_data_path)

    # 2. Read config for model checkpoint, batch_size, max_length
    model_checkpoint = nlp_config.get("model_checkpoint", "ProsusAI/finbert")
    batch_size = int(nlp_config.get("batch_size", 16))
    max_length = int(nlp_config.get("max_length", 512))

    # 3. Initialize sentiment pipeline
    sentiment_pipeline = load_sentiment_pipeline(model_checkpoint, batch_size, max_length)

    # 4. Compute sentiment scores
    text_column = nlp_config.get("text_column", "headline")
    labels, scores = compute_sentiment_scores(df, text_column, sentiment_pipeline)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    # 5. Convert label + score to a numeric value
    df["sentiment_value"] = [
        map_label_to_value(lbl, scr) for lbl, scr in zip(labels, scores)
    ]

    # 6. Derive signals from sentiment (simple threshold example)
    buy_threshold = float(nlp_config.get("buy_threshold", 0.2))
    sell_threshold = float(nlp_config.get("sell_threshold", -0.2))

    def derive_signal(value):
        if value > buy_threshold:
            return "BUY"
        elif value < sell_threshold:
            return "SELL"
        else:
            return "HOLD"

    df["trade_signal"] = df["sentiment_value"].apply(derive_signal)

    # 7. Save results
    output_path = nlp_config.get("output_path", "data/interim/nlp_signals.csv")
    df.to_csv(output_path, index=False)
    logging.info("NLP pipeline complete. Signals saved to: %s", output_path)
