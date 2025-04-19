# indicators/ema.py

import pandas as pd

def calculate_ema(df: pd.DataFrame, period: int = 20, column: str = "close") -> pd.Series:
    """
    محاسبه EMA برای یک ستون خاص از DataFrame
    :param df: دیتافریم کندل‌ها
    :param period: تعداد دوره‌ها برای EMA (پیش‌فرض 20)
    :param column: ستون مورد نظر (مثلاً 'close')
    :return: سری EMA
    """
    return df[column].ewm(span=period, adjust=False).mean()
