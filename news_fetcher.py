import requests
from data.data_manager import insert_news
from config import SYMBOLS
from datetime import datetime, timedelta
import re
from ai.news_sentiment_ai import analyze_sentiment
import mysql.connector
from config import MYSQL_CONFIG

API_KEY = "da4aff9ddef8f6534f149d3803e2e0420958c5f4"

def fetch_news():
    print("📰 در حال دریافت اخبار...")
    # فقط اخبار ۷ روز اخیر رو بگیر
    since_date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true&kind=news&published_since={since_date}"

    try:
        response = requests.get(url)
        data = response.json()
        print(f"اخبار دریافت‌شده: {data['results']}")
        
        if "results" not in data:
            print("⚠️ هیچ نتیجه‌ای یافت نشد.")
            return

        # اتصال به دیتابیس
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        for item in data["results"]:
            title = item.get("title", "")
            content = item.get("content", title)
            if not title:
                continue

            source = item.get("domain", "unknown")
            published_at = item.get("published_at", datetime.utcnow().isoformat())
            # تبدیل published_at به فرمت datetime
            published_at_dt = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%S%z')

            # چک کردن خبر تکراری
            check_query = """
            SELECT COUNT(*) FROM news
            WHERE symbol = %s AND title = %s AND published_at = %s
            """
            for symbol in SYMBOLS:
                pattern = r'\b' + re.escape(symbol.replace("USDT", "").upper()) + r'\b'
                if re.search(pattern, title.upper()):
                    cursor.execute(check_query, (symbol, title, published_at))
                    result = cursor.fetchone()
                    if result[0] > 0:  # اگه خبر قبلاً ذخیره شده
                        print(f"⚠️ خبر تکراری برای {symbol} - {title[:40]}...")
                        continue

                    # محاسبه sentiment_score
                    sentiment_score = 0.0
                    if content and content.strip():
                        sentiment_score = analyze_sentiment(content)
                    else:
                        print(f"⚠️ محتوا یا عنوان خالی برای خبر: {title}")

                    # ذخیره خبر
                    insert_news(symbol, title, source, published_at_dt, content, None)

                    # فیلتر اخبار مهم
                    important_keywords = ['partnership', 'regulation', 'adoption', 'upgrade', 'halving']
                    is_important = any(keyword in title.lower() for keyword in important_keywords)
                    print(f"✅ خبر ذخیره شد برای {symbol} - {title[:40]}... {'(مهم)' if is_important else ''}")

        conn.commit()

    except Exception as e:
        print(f"❌ خطا در دریافت اخبار: {e}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    fetch_news()