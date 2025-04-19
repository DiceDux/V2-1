# data/data_manager.py
import mysql.connector
from datetime import datetime
import pandas as pd
from config import MYSQL_CONFIG
import mysql.connector
from sqlalchemy import create_engine

engine = create_engine(f"mysql+mysqlconnector://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")
def get_connection():
    return mysql.connector.connect(**MYSQL_CONFIG)

def save_candles_to_db(symbol, df: pd.DataFrame):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        for _, row in df.iterrows():
            timestamp = int(row["timestamp"])  # ✅ درست شد
            cursor.execute("""
                INSERT INTO candles (symbol, timestamp, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE open=VALUES(open), high=VALUES(high),
                low=VALUES(low), close=VALUES(close), volume=VALUES(volume)
            """, (symbol, timestamp, row['open'], row['high'], row['low'], row['close'], row['volume']))

        conn.commit()
        print(f"✅ داده‌های کندل {symbol} ذخیره شدند.")

    except Exception as e:
        print("❌ خطا در ذخیره کندل:", e)
    finally:
        if conn.is_connected():
            conn.close()

def save_features_to_db(symbol, interval, features: dict):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        timestamp = int(datetime.utcnow().timestamp())

        cursor.execute("""
            INSERT INTO features (symbol, `interval`, timestamp, rsi, ema20, ch, atr, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                rsi=VALUES(rsi),
                ema20=VALUES(ema20),
                ch=VALUES(ch),
                atr=VALUES(atr),
                volume=VALUES(volume)
        """, (
            symbol, interval, timestamp,
            features.get("rsi"),
            features.get("ema20"),
            features.get("ch"),
            features.get("atr"),
            features.get("volume")
        ))

        conn.commit()
        print(f"✅ ویژگی‌های {symbol} با interval {interval} ذخیره شدند.")

    except Exception as e:
        print("❌ خطا در ذخیره ویژگی‌ها:", e)
    finally:
        if conn.is_connected():
            conn.close()

def get_features_from_db(symbol, interval):
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT rsi, ema20, ch, atr, volume
            FROM features
            WHERE symbol=%s AND `interval`=%s
            ORDER BY timestamp DESC
            LIMIT 1
        """, (symbol, interval))

        row = cursor.fetchone()
        return row if row else None

    except Exception as e:
        print("❌ خطا در گرفتن ویژگی‌ها از DB:", e)
        return None
    finally:
        if conn.is_connected():
            conn.close()

def insert_balance_to_db(symbol, balance):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()

    now = datetime.utcnow().isoformat()
    cursor.execute("""
        INSERT INTO balance (symbol, balance, updated_at)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            balance = VALUES(balance),
            updated_at = VALUES(updated_at)
    """, ("WALLET", balance, now))  # فقط یک رکورد با symbol = WALLET

    conn.commit()
    conn.close()

def save_trade_record(symbol, action, price, balance, confidence, profit=None):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        now = datetime.utcnow().isoformat()
        sql = """
            INSERT INTO trades (symbol, action, entry_price, confidence, balance, profit, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (symbol, action.upper(), price, confidence, balance, profit, now)

        cursor.execute(sql, values)
        conn.commit()
        print(f"📥 ترید برای {symbol} ذخیره شد.")

    except Exception as e:
        print("❌ خطا در save_trade_record:", e)

    finally:
        if conn.is_connected():
            conn.close()

# ➕ تابع جدید برای گرفتن داده‌های کندل و ذخیره آن‌ها در دیتابیس
from data.fetch_coinex import fetch_candles

def get_candle_data(symbol: str, limit: int = 1000):
    try:
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        df = pd.read_sql(query, engine, params=(symbol, limit))

        # شرط کاهش لیمیت اگر داده کم باشه
        if len(df) < 900 and limit == 1000:
            print(f"⚠️ فقط {len(df)} کندل برای {symbol} یافت شد. تلاش مجدد با ۵۰۰ کندل...")
            df = pd.read_sql(query, engine, params=(symbol, 500))  # اینجا conn به engine تغییر کرد

        if df.empty:
            print(f"⚠️ دیتافریم خالی برای {symbol}")
            return pd.DataFrame()

        df = df.sort_values(by="timestamp")
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        print(f"❌ خطا در دریافت کندل از دیتابیس برای {symbol}: {e}")
        return pd.DataFrame()

# ➕ تابع ساده برای ذخیره ویژگی‌ها در دیتابیس (برای main_ml)
def save_features(symbol, features):
    from config import TIMEFRAME_MINUTES
    interval = f"{TIMEFRAME_MINUTES}min"
    save_features_to_db(symbol, interval, features)

# داخل data_manager.py

def has_open_trade(symbol):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM open_trades WHERE symbol = %s", (symbol,))
    row = cursor.fetchone()
    conn.close()
    return row if row else None

def insert_open_trade(symbol, action, price):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO open_trades (symbol, action, entry_price, timestamp)
        VALUES (%s, %s, %s, %s)
    """, (symbol, action, price, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def close_open_trade(symbol):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM open_trades WHERE symbol = %s", (symbol,))
    conn.commit()
    conn.close()

def has_trade_in_current_candle(symbol, current_candle_time):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        SELECT COUNT(*) FROM trades
        WHERE symbol = %s AND candle_time = %s
    """
    cursor.execute(query, (symbol, current_candle_time))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0

def insert_trade_with_candle(symbol, action, price, confidence, candle_time, mode, profit=None, exit_price=None, quantity=None):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO trades (
            symbol, action, entry_price, confidence, timestamp, mode,
            candle_time, profit, exit_price, quantity
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        symbol, action, price, confidence, timestamp, mode,
        candle_time, profit, exit_price, quantity
    ))

    conn.commit()
    conn.close()


def insert_position(symbol, action, entry_price, quantity, tp_price, sl_price, tp_step=1, last_price=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO positions (symbol, action, entry_price, quantity, tp_price, sl_price, tp_step, last_price)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (symbol, action, entry_price, quantity, tp_price, sl_price, tp_step, last_price))
    conn.commit()
    conn.close()

def get_position(symbol):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM positions WHERE symbol = %s LIMIT 1", (symbol,))
    row = cursor.fetchone()
    conn.close()
    return row

def delete_position(symbol):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM positions WHERE symbol = %s", (symbol,))
    conn.commit()
    conn.close()

def get_wallet_balance():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM wallet LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 0.0

def update_wallet_balance(new_balance):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    cursor.execute("""
        UPDATE wallet SET balance = %s, updated_at = %s WHERE id = 1
    """, (new_balance, now))
    conn.commit()
    conn.close()

def update_position_trailing(symbol, new_tp, new_sl, new_step, last_price):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE positions
        SET tp_price = %s, sl_price = %s, tp_step = %s, last_price = %s
        WHERE symbol = %s
    """, (new_tp, new_sl, new_step, last_price, symbol))
    conn.commit()
    conn.close()

def insert_news(symbol, title, source, published_at, content=None, sentiment_score=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO news (symbol, title, source, published_at, content, sentiment_score)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (symbol, title, source, published_at, content, sentiment_score))
    conn.commit()
    conn.close()

def get_sentiment_from_db(symbol, timestamp):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT AVG(sentiment_score) as avg_sentiment
        FROM news
        WHERE symbol = %s
          AND published_at <= FROM_UNIXTIME(%s)
          AND published_at >= FROM_UNIXTIME(%s - 60*60*4)
    """, (symbol.replace("USDT", ""), timestamp, timestamp))

    row = cursor.fetchone()
    conn.close()
    return row["avg_sentiment"] if row and row["avg_sentiment"] is not None else 0.0

def get_recent_news_texts(symbol, timestamp, hours=6):
    from datetime import datetime, timedelta

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    start_dt = datetime.utcfromtimestamp(timestamp - hours * 3600).isoformat()
    cursor.execute("""
        SELECT COALESCE(content, title) as text FROM news
        WHERE symbol = %s AND published_at >= %s
        ORDER BY published_at DESC
    """, (symbol, start_dt))

    rows = cursor.fetchall()
    conn.close()

    combined_text = ' '.join([row["title"] for row in rows if "title" in row])
    return combined_text

def get_index_value(name, timestamp):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT value FROM market_indices
        WHERE name = %s AND timestamp <= %s
        ORDER BY timestamp DESC LIMIT 1
    """, (name, timestamp))
    row = cursor.fetchone()
    conn.close()
    return row['value'] if row else None
