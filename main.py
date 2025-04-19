# main.py

from data.fetch_coinex import fetch_candles
from ai.ai_engine import analyze_market
from trading.simulation import simulate_trade
from config import SYMBOLS

def run():
    print("🚀 DiceDux Trade در حال اجراست...\n")

    for symbol in SYMBOLS:
        print(f"🔎 بررسی {symbol} ...")
        df = fetch_candles(symbol)
        if df.empty:
            print(f"⚠️ دریافت داده برای {symbol} ناموفق بود.\n")
            continue

        signal = analyze_market(df)
        price = df["close"].iloc[-1]

        print(f"📈 سیگنال: {signal['action'].upper()} | قیمت: {price} | اعتماد: {signal['confidence']}")

        if signal["action"] in ["buy", "sell"]:
            result = simulate_trade(symbol, signal["action"], price)
            print(f"✅ معامله {signal['action']} ثبت شد | موجودی جدید: {result['balance']}\n")
        else:
            print("⏸️ فعلاً در حالت HOLD هستیم.\n")

if __name__ == "__main__":
    run()
