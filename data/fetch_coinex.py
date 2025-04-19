import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
import pandas as pd
import time
from datetime import datetime
from config import COINEX_BASE_URL, CANDLE_HISTORY_LIMIT, TIMEFRAME_MINUTES, SYMBOLS
from data.data_manager import save_candles_to_db

def convert_timeframe(minutes: int) -> str:
    mapping = {
        1: "1min", 3: "3min", 5: "5min", 15: "15min", 30: "30min",
        60: "1hour", 120: "2hour", 240: "4hour", 360: "6hour",
        720: "12hour", 1440: "1day"
    }
    return mapping.get(minutes, "4hour")

def fetch_candles(symbol: str, limit: int = 2):
    url = f"{COINEX_BASE_URL}/v1/market/kline"
    period = convert_timeframe(TIMEFRAME_MINUTES)

    params = {
        "market": symbol,
        "type": period,
        "limit": limit
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        if data['code'] != 0:
            print(f"❌ خطا در دریافت داده برای {symbol}: {data['message']}")
            return pd.DataFrame()

        rows = data['data']
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "close", "high", "low", "volume", "amount"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["timestamp"] = df["timestamp"].apply(lambda x: int(x.timestamp()))

        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # فقط کندل‌های جدید رو ذخیره می‌کنیم
        save_candles_to_db(symbol, df)
        print(f"✅ کندل‌های {symbol} به‌روز شد | آخرین timestamp: {df['timestamp'].iloc[-1]}")
        return df

    except Exception as e:
        print(f"❌ خطا در دریافت داده برای {symbol}: {e}")
        return pd.DataFrame()

def run_live_candle_fetcher():
    while True:
        for symbol in SYMBOLS:
            print(f"🔄 به‌روزرسانی کندل {symbol}")
            fetch_candles(symbol, limit=1)  # فقط آخرین کندل
        time.sleep(5)  # تأخیر 5 ثانیه برای هماهنگی بهتر

if __name__ == "__main__":
    mode = "live"  # "initial" or "recovery" or "live"

    if mode == "initial":
        for symbol in SYMBOLS:
            fetch_candles(symbol, limit=1000)

    elif mode == "recovery":
        for symbol in SYMBOLS:
            fetch_candles(symbol, limit=200)

    elif mode == "live":
        run_live_candle_fetcher()