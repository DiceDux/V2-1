# price_updater.py
import time
import mysql.connector
import asyncio
import websockets
import threading
from config import MYSQL_CONFIG

def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def update_prices():
    while True:
        try:
            conn = get_connection()
            cursor = conn.cursor(dictionary=True)

            cursor.execute("SELECT * FROM positions")
            positions = cursor.fetchall()

            for pos in positions:
                symbol = pos['symbol']
                last_tp = pos['tp_price']
                last_sl = pos['sl_price']
                step = pos['tp_step']
                entry = pos['entry_price']

                # گرفتن آخرین قیمت از جدول کندل‌ها
                cursor.execute("""
                    SELECT close, timestamp FROM candles
                    WHERE symbol = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol,))
                result = cursor.fetchone()
                if not result:
                    print(f"⚠️ کندلی برای {symbol} یافت نشد.")
                    continue

                current_price = float(result['close'])
                latest_timestamp = result['timestamp']
                profit_percent = ((current_price - entry) / entry) * 100

                print(f"🟡 بررسی {symbol} | قیمت: {current_price} | SL: {last_sl} | TP: {last_tp} | Timestamp: {latest_timestamp}")

                # آپدیت قیمت و سود زنده
                cursor.execute("""
                    UPDATE positions SET last_price = %s, live_profit = %s WHERE symbol = %s
                """, (current_price, profit_percent, symbol))

                # بررسی Trailing TP/SL
                if current_price >= last_tp:
                    new_tp = round(last_tp * 1.03, 8)  # تغییر به 3 درصد
                    new_sl = last_tp  # SL جدید برابر TP قبلی
                    new_step = step + 1

                    cursor.execute("""
                        UPDATE positions
                        SET tp_price = %s, sl_price = %s, tp_step = %s
                        WHERE symbol = %s
                    """, (new_tp, new_sl, new_step, symbol))

                    print(f"[TP STEP] {symbol} reached TP. New TP: {new_tp}, New SL: {new_sl}")

            conn.commit()

        except Exception as e:
            print(f"❌ Error in price_updater: {e}")

        finally:
            conn.close()

        time.sleep(2)  # تأخیر 2 ثانیه برای هماهنگی با fetch_coinex

async def check_prices_and_notify():
    while True:
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM positions")
            positions = cursor.fetchall()

            for pos in positions:
                symbol = pos['symbol']
                sl = float(pos['sl_price'])

                # گرفتن آخرین قیمت لحظه‌ای از جدول کندل‌ها
                cursor.execute("""
                    SELECT close FROM candles
                    WHERE symbol = %s
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol,))
                result = cursor.fetchone()
                if not result:
                    continue

                current_price = float(result['close'])

                if current_price <= sl:
                    print(f"🟥 SL فعال شد برای {symbol} | قیمت: {current_price} <= SL: {sl} | زمان: {time.time()}")
                    async with websockets.connect("ws://localhost:5678") as websocket:
                        await websocket.send(f"SELL:{symbol}")
                        print(f"📤 ارسال SELL برای SL {symbol}")

        except Exception as e:
            print("❌ خطا در price_updater:", e)

        finally:
            if conn.is_connected():
                conn.close()

        await asyncio.sleep(0.5)  # کاهش زمان به 0.5 ثانیه برای واکنش سریع‌تر

if __name__ == "__main__":
    threading.Thread(target=update_prices, daemon=True).start()
    asyncio.run(check_prices_and_notify())