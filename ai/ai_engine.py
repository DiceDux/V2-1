# ai/ai_engine.py

import pandas as pd
from indicators.rsi import calculate_rsi
from indicators.ema import calculate_ema
from indicators.ch import calculate_ch
from indicators.atr import calculate_atr
from indicators.volume import detect_volume_spike
from indicators.patterns import detect_double_bottom, detect_double_top

def analyze_market(df: pd.DataFrame) -> dict:
    """
    تحلیل کامل بازار بر اساس اندیکاتورها و الگوها
    خروجی: تصمیم خرید / فروش یا نگه‌داشتن
    """
    df = df.copy()

    # محاسبه اندیکاتورها
    df["rsi"] = calculate_rsi(df)
    df["ema20"] = calculate_ema(df, period=20)
    df["ch"] = calculate_ch(df)
    df["atr"] = calculate_atr(df)
    df["volume_spike"] = detect_volume_spike(df)
    df["double_bottom"] = detect_double_bottom(df)
    df["double_top"] = detect_double_top(df)

    # کندل آخر
    last = df.iloc[-1]

    # قوانین خرید:
    buy_conditions = [
        last["rsi"] < 30,
        last["close"] > last["ema20"],
        last["volume_spike"] == True,
        last["double_bottom"] == True
    ]

    # قوانین فروش:
    sell_conditions = [
        last["rsi"] > 70,
        last["close"] < last["ema20"],
        last["volume_spike"] == True,
        last["double_top"] == True
    ]

    if all(buy_conditions):
        return {"action": "buy", "confidence": 0.9}
    elif all(sell_conditions):
        return {"action": "sell", "confidence": 0.9}
    else:
        return {"action": "hold", "confidence": 0.5}
