import mysql.connector
import numpy as np
from transformers import pipeline
import datetime
from config import MYSQL_CONFIG
import pandas as pd

# مدل NLP برای تحلیل احساسات
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_fundamental_score_from_db(symbol: str, timestamp: int, volume: float = 0.0) -> dict:
    """
    تحلیل احساسات خبرها و ترکیب با حجم معاملات برای امتیاز فاندامنتال
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        time_window = 6 * 3600  # ۶ ساعت قبل و بعد (طبق پیشنهاد قبلی)
        from_ts = timestamp - time_window
        to_ts = timestamp + time_window

        query = """
        SELECT COALESCE(content, title) as text, source, published_at, sentiment_score
        FROM news
        WHERE symbol = %s AND published_at BETWEEN FROM_UNIXTIME(%s) AND FROM_UNIXTIME(%s)
        ORDER BY published_at DESC
        LIMIT 10
        """
        cursor.execute(query, (symbol, from_ts, to_ts))
        results = cursor.fetchall()
        texts = [(row[0][:512], row[1], row[2], row[3]) for row in results if row[0]]

        news_score = 0.0
        print(f"اخبار خوانده‌شده از دیتابیس برای {symbol}: {texts}")
        if texts:
            scores = []
            for text, source, published_at, sentiment_score in texts:
                try:
                    # اگه sentiment_score موجود بود، ازش استفاده کن
                    if sentiment_score is not None:
                        score = sentiment_score
                    else:
                        # در غیر این صورت، احساسات رو محاسبه کن
                        result = sentiment_pipeline(text)[0]
                        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']

                    # وزن‌دهی بر اساس منبع و تازگی خبر
                    weight = 1.5 if source in ['coindesk', 'cointelegraph'] else 1.0
                    time_diff = (timestamp - int(published_at.timestamp())) / 3600
                    time_weight = max(1.0 - time_diff / 24, 0.5)  # اخبار جدیدتر وزن بیشتری دارن
                    scores.append(score * weight * time_weight)
                except Exception as e:
                    print(f"⚠️ خطا در پردازش متن خبر: {e}")
                    continue
            news_score = float(np.mean(scores)) if scores else 0.0

        # امتیاز حجم معاملات (نرمال‌سازی ساده)
        volume_score = min(volume / 1e7, 1.0) if volume > 0 else 0.0  # فرض: مقیاس به 10 میلیون

        # امتیاز فاندامنتال ترکیبی
        fundamental_score = 0.6 * news_score + 0.4 * volume_score

        return {
            'fundamental_score': fundamental_score,
            'news_score': news_score,
            'volume_score': volume_score
        }

    except Exception as e:
        print("⚠️ خطا در اتصال به دیتابیس یا پردازش خبر:", e)
        return {
            'fundamental_score': 0.0,
            'news_score': 0.0,
            'volume_score': 0.0
        }
    finally:
        if 'conn' in locals() and conn.is_connected():
            conn.close()

def analyze_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    """اضافه کردن امتیازات فاندامنتال به دیتافریم"""
    df = df.copy()
    fundamental_scores = []
    news_scores = []
    volume_scores = []

    for _, row in df.iterrows():
        try:
            scores = get_fundamental_score_from_db(row['symbol'], row['timestamp'], row.get('volume', 0.0))
            fundamental_scores.append(scores['fundamental_score'])
            news_scores.append(scores['news_score'])
            volume_scores.append(scores['volume_score'])
        except Exception as e:
            print(f"⚠️ خطا در تحلیل فاندامنتال برای {row['symbol']}: {e}")
            fundamental_scores.append(0.0)
            news_scores.append(0.0)
            volume_scores.append(0.0)
    
    df['fundamental_score'] = fundamental_scores
    df['news_score'] = news_scores
    df['volume_score'] = volume_scores
    return df