import time
import yfinance as yf
import requests
import mysql.connector
from config import MYSQL_CONFIG
from datetime import datetime
import pandas as pd

INDEXES = {
    "BTC.D": "https://api.coingecko.com/api/v3/global",
    "USDT.D": "https://api.alternative.me/fng/",
    "SPX": "^GSPC",
    "DXY": "UUP"
}

def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def fetch_btc_dominance():
    try:
        r = requests.get(INDEXES["BTC.D"])
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception as e:
        print("❌ BTC.D fetch error:", e)
        return None

def fetch_usdt_dominance():
    try:
        r = requests.get(INDEXES["USDT.D"])
        return float(r.json()["data"][0]["value"])
    except Exception as e:
        print("❌ USDT.D fetch error:", e)
        return None

def fetch_yfinance_index(ticker):
    try:
        data = yf.download(ticker, period="1d", interval="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else None
    except Exception as e:
        print(f"❌ yfinance fetch error for {ticker}:", e)
        return None

def save_index(name, value):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = int(time.time())
        cursor.execute("""
            INSERT INTO market_indices (name, timestamp, value, index_name)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE value = VALUES(value)
        """, (name, timestamp, value, name))  # ← index_name هم ست می‌شه
        conn.commit()
        conn.close()
        print(f"✅ Saved {name}: {value}")
    except Exception as e:
        print(f"❌ Error saving {name}:", e)

def fetch_and_store_yfinance_indexes():
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=10)
    symbols = {
        "SPX": "^GSPC",
        "DXY": "DX-Y.NYB"
    }

    for name, yf_symbol in symbols.items():
        df = yf.download(yf_symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='1d')
        if df.empty:
            print(f"⚠️ داده‌ای برای {name} نبود")
            continue

        for ts, row in df.iterrows():
            unix_ts = int(pd.Timestamp(ts).timestamp())
            value = float(row["Close"])

            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_indices (name, timestamp, value, index_name)
                VALUES (%s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE value = VALUES(value)
            """, (name, unix_ts, value, name))
            conn.commit()
            cursor.close()
            conn.close()

        print(f"✅ {name} آپدیت شد: {len(df)} روز")

fetch_and_store_yfinance_indexes()


if __name__ == "__main__":
    btc_d = fetch_btc_dominance()
    usdt_d = fetch_usdt_dominance()
    spx = fetch_yfinance_index(INDEXES["SPX"])
    dxy = fetch_yfinance_index(INDEXES["DXY"])

    if btc_d is not None:
        save_index("BTC.D", btc_d)
    if usdt_d is not None:
        save_index("USDT.D", usdt_d)
    if spx is not None:
        save_index("SPX", spx)
    if dxy is not None:
        save_index("DXY", dxy)

    print("✅ All market indices updated.")
