# indicators/patterns.py

import pandas as pd

def detect_double_bottom(df: pd.DataFrame) -> pd.Series:
    """
    تشخیص الگوی Double Bottom در دیتافریم
    الگویی ساده برای تست: دوتا کف نزدیک با سقف بینشون
    """
    signals = [False] * len(df)
    for i in range(2, len(df) - 1):
        p1 = df["low"].iloc[i-2]
        p2 = df["low"].iloc[i]
        mid = df["high"].iloc[i-1]
        if abs(p1 - p2) / p1 < 0.02 and mid > p1 * 1.01:
            signals[i] = True
    return pd.Series(signals, index=df.index)

def detect_double_top(df: pd.DataFrame) -> pd.Series:
    """
    تشخیص الگوی Double Top
    دوتا سقف نزدیک با کف بینشون
    """
    signals = [False] * len(df)
    for i in range(2, len(df) - 1):
        p1 = df["high"].iloc[i-2]
        p2 = df["high"].iloc[i]
        mid = df["low"].iloc[i-1]
        if abs(p1 - p2) / p1 < 0.02 and mid < p1 * 0.99:
            signals[i] = True
    return pd.Series(signals, index=df.index)
