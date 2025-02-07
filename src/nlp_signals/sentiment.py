import logging
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def load_sentiment_pipeline(model_checkpoint: str):
    """
    Loads a sentiment classification pipeline given a model checkpoint.
    Example: 'ProsusAI/finbert', 'FinGPT-Base', etc.
    """
    logging.info(f"Loading sentiment pipeline from checkpoint: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def compute_sentiment_scores(df: pd.DataFrame, text_column: str, sentiment_pipeline):
    """
    Applies the sentiment_pipeline to the specified text_column in df.
    Returns a tuple of two lists: (labels, scores).
    """
    logging.info(f"Computing sentiment for column: {text_column}")
    text_data = df[text_column].fillna("").tolist()
    results = sentiment_pipeline(text_data)
    labels = [res["label"] for res in results]
    scores = [res["score"] for res in results]
    return labels, scores

def map_label_to_value(label: str, score: float) -> float:
    """
    Converts a label + score to a single numeric sentiment value.
      - POSITIVE => +score
      - NEGATIVE => -score
      - NEUTRAL  =>  0
    Adjust logic here if your model uses different label names.
    """
    if label.upper() == "POSITIVE":
        return score
    elif label.upper() == "NEGATIVE":
        return -score
    else:
        return 0.0
