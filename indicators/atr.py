# indicators/atr.py

import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    محاسبه ATR (Average True Range)
    :param df: دیتافریم کندل‌ها با ستون‌های high، low، close
    :param period: تعداد دوره‌ها (پیش‌فرض 14)
    :return: سری ATR
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr
