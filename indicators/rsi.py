# indicators/rsi.py

import pandas as pd

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    محاسبه RSI برای DataFrame
    :param df: دیتافریم کندل‌ها با ستون 'close'
    :param period: دوره زمانی RSI (پیش‌فرض 14)
    :return: سری RSI
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
