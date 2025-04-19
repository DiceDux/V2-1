# indicators/volume.py

import pandas as pd

def calculate_volume_stats(df: pd.DataFrame, period: int = 20):
    """
    محاسبه میانگین حجم و انحراف معیار برای بررسی حجم‌های غیرعادی
    :param df: دیتافریم کندل‌ها با ستون 'volume'
    :param period: تعداد دوره‌ها برای محاسبه
    :return: میانگین و انحراف معیار حجم
    """
    avg_volume = df["volume"].rolling(window=period).mean()
    std_volume = df["volume"].rolling(window=period).std()
    return avg_volume, std_volume

def detect_volume_spike(df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> pd.Series:
    """
    تشخیص پرش ناگهانی در حجم (Volume Spike)
    :param df: دیتافریم کندل‌ها
    :param period: دوره میانگین‌گیری
    :param multiplier: ضریب پرش (مثلاً ۲ برابر میانگین)
    :return: سری True/False
    """
    avg_volume, std_volume = calculate_volume_stats(df, period)
    return df["volume"] > (avg_volume + multiplier * std_volume)
