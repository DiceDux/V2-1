# main_ml.py
import time
import subprocess
import asyncio
import websockets
import threading
import os
import socket


from ai.ai_model_runner import predict_signal_from_model
from trading.simulation import simulate_trade
from data.data_manager import get_candle_data, save_features
from data.data_manager import get_position, insert_position, delete_position
from config import SYMBOLS
from trading.trade_status import get_trade_status
from config import TRADE_MODE, TIMEFRAME_MINUTES
from data.data_manager import has_trade_in_current_candle, insert_trade_with_candle
from data.data_manager import insert_balance_to_db
from data.data_manager import update_position_trailing

async def websocket_handler(websocket):
    try:
        async for message in websocket:
            print(f"📩 پیام WebSocket دریافت شد: {message}")
            # پردازش مستقیم پیام‌ها
            if message.startswith("SELL:"):
                symbol = message.split(":")[1]
                print(f"🚨 دستور فروش فوری دریافت شد برای {symbol}")
                position = get_position(symbol)
                if position:
                    entry_price = float(position["entry_price"])
                    price = float(position["last_price"])
                    profit = price - entry_price
                    profit_percent = (profit / entry_price) * 100
                    result = simulate_trade(symbol, "sell", price, confidence=0.0)
                    delete_position(symbol)
                    candle_time = int((int(time.time()) // (TIMEFRAME_MINUTES * 60)) * (TIMEFRAME_MINUTES * 60))
                    insert_trade_with_candle(
                        symbol, "sell", price, 0.0, candle_time, TRADE_MODE,
                        profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                    )
                    insert_balance_to_db("WALLET", result["balance"])
                    print(f"✅ فروش فوری انجام شد برای {symbol} | سود: {profit_percent:.2f}% | موجودی: {result['balance']}")
                else:
                    print(f"⚠️ پوزیشنی برای {symbol} وجود ندارد. فروش انجام نشد.")
    except Exception as e:
        print(f"❌ خطا در WebSocket: {e}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if is_port_in_use(5678):
    print("❌ خطا: پورت 5678 در حال استفاده است. برنامه متوقف می‌شود.")
    exit(1)

async def start_ws_server_async():
    async with websockets.serve(websocket_handler, "0.0.0.0", 5678):
        print("🌐 WebSocket server فعال شد روی پورت 5678")
        await asyncio.Future()  # اجرای بی‌نهایت

def start_ws_server():
    asyncio.run(start_ws_server_async())

# شروع سرور WebSocket در ترد جداگانه
if not is_port_in_use(5678):
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()
    print("🌐 WebSocket server فعال شد روی پورت 5678")
else:
    print("⚠️ WebSocket server قبلاً فعال شده. از راه‌اندازی مجدد جلوگیری شد.")
  

def run_with_ml():
    print("SYMBOLS:", SYMBOLS)  # لاگ برای SYMBOLS
    if not SYMBOLS:
        print("❌ خطا: SYMBOLS خالی است!")
        return

    if get_trade_status() == "paused":
        print("⏸ ربات در حالت توقف قرار دارد.")
        return 

    print("🚀 DiceDux ML در حال اجرا با مدل یادگیرنده...\n")

    for symbol in SYMBOLS:
        print(f"🔎 تحلیل ML برای {symbol}")
        df = get_candle_data(symbol)
        print(f"داده‌های کندل برای {symbol}: {df.shape}")  # لاگ برای داده‌ها

        if df.empty or len(df) < 50:
            print("⚠️ داده کافی برای تحلیل وجود ندارد.\n")
            continue

        try:
            signal = predict_signal_from_model(df, symbol=symbol, interval=f"{TIMEFRAME_MINUTES}min", verbose=True)
            signal["action"] = str(signal["action"]).lower().strip("[]' ")
            price = df["close"].iloc[-1]
            print(f"🤖 سیگنال مدل: {signal['action'].upper()} | قیمت: {price} | اعتماد: {signal['confidence']}")
        except Exception as e:
            print(f"❌ خطا در پیش‌بینی برای {symbol}: {e}")
            continue
        
        # ✅ بررسی حداقل اعتماد
        if signal["confidence"] < 0.70:
            print("⚠️ اعتماد کافی وجود ندارد. ترید انجام نمی‌شود.\n")
            continue

        position = get_position(symbol)
        save_features(symbol, signal['features'])
        print(f"📰 احساسات خبری: {signal['features'].get('news_sentiment', 0)}")

        # محاسبه timestamp کندل فعلی
        candle_time = int((int(time.time()) // (TIMEFRAME_MINUTES * 60)) * (TIMEFRAME_MINUTES * 60))

        # جلوگیری از ترید تکراری در همین کندل
        if has_trade_in_current_candle(symbol, candle_time):
            print(f"⛔️ تریدی برای {symbol} در همین کندل ثبت شده. عبور می‌کنیم.\n")
            continue

        if signal["action"] == "buy":
            if position:
                print(f"🚫 پوزیشن باز برای {symbol} وجود دارد. خرید مجدد مجاز نیست.")
                continue
            else:
                result = simulate_trade(symbol, "buy", price, signal["confidence"])
                insert_trade_with_candle(symbol, "buy", price, signal["confidence"], candle_time, TRADE_MODE)
                insert_balance_to_db("WALLET", result["balance"])
                print(f"✅ خرید انجام شد | موجودی: {result['balance']}")


        # اگر پوزیشن باز وجود دارد، بررسی فروش (چه با سیگنال چه فقط سود)
        if position:
            entry_price = float(position["entry_price"])
            quantity = float(position["quantity"])
            tp_price = float(position["tp_price"])
            sl_price = float(position["sl_price"])
            tp_step = int(position["tp_step"])
            last_price = float(position["last_price"])
            profit = price - entry_price
            profit_percent = (profit / entry_price) * 100

            if signal["action"] == "sell" and signal["confidence"] >= 0.70:
                result = simulate_trade(symbol, "sell", price, signal["confidence"])
                delete_position(symbol)
                insert_trade_with_candle(
                    symbol, "sell", price, 0.0, candle_time, TRADE_MODE,
                    profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                )
                insert_balance_to_db("WALLET", result["balance"])
                print(f"✅ فروش انجام شد با سیگنال مدل | سود: {profit_percent:.2f}% | موجودی: {result['balance']}")
     
            # آپدیت پله‌ای TP/SL
            if price >= tp_price:
                new_tp_step = tp_step + 1
                new_tp_price = entry_price * (1 + 0.05 * new_tp_step)
                new_sl_price = tp_price
                update_position_trailing(symbol, new_tp_price, new_sl_price, new_tp_step, price)
                print(f"📈 TP رسید | مرحله بعدی TP: {new_tp_price:.2f}, SL جدید: {new_sl_price:.2f}")

            # فعال شدن SL
            elif price <= sl_price:
                result = simulate_trade(symbol, "sell", price, confidence=0.0)
                delete_position(symbol)
                insert_trade_with_candle(
                    symbol, "sell", price, 0.0, candle_time, TRADE_MODE,
                    profit_percent, exit_price=result.get("exit_price"), quantity=result.get("quantity")
                )
                insert_balance_to_db("WALLET", result["balance"])
                print(f"⛔️ SL فعال شد | فروش با قیمت: {price:.2f} | موجودی: {result['balance']:.2f}")
                continue

            else:
                print(f"⏳ فروش انجام نشد. سود فعلی: {profit_percent:.2f}٪ | اعتماد مدل: {signal['confidence']}\n")

if __name__ == "__main__":
    while True:
        try:
            run_with_ml()
        except Exception as e:
            print(f"❌ خطای غیرمنتظره در اجرای run_with_ml: {e}")
        print("⏳ در حال صبر برای تحلیل بعدی...\n")
        time.sleep(60 * 5)