# indicators/ch.py

import pandas as pd

def calculate_ch(df: pd.DataFrame) -> pd.Series:
    """
    محاسبه CH (Candle Height) یا دامنه کندل (high - low)
    :param df: دیتافریم کندل‌ها با ستون‌های high و low
    :return: سری CH
    """
    return df["high"] - df["low"]
