import requests
from data.data_manager import insert_news
from config import SYMBOLS
from datetime import datetime
import re
from ai.news_sentiment_ai import analyze_sentiment

API_KEY = "da4aff9ddef8f6534f149d3803e2e0420958c5f4"

def fetch_news():
    print("📰 در حال دریافت اخبار...")
    url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true&kind=news"

    try:
        response = requests.get(url)
        data = response.json()

        if "results" not in data:
            print("⚠️ هیچ نتیجه‌ای یافت نشد.")
            return

        for item in data["results"]:
            title = item.get("title", "")
            content = item.get("content", title)
            if not title:
                continue

            source = item.get("domain", "unknown")
            published_at = item.get("published_at", datetime.utcnow().isoformat())
            sentiment_score = analyze_sentiment(content)

            # فیلتر اخبار مهم (مثلاً فقط اخبار با کلمات کلیدی خاص)
            important_keywords = ['partnership', 'regulation', 'adoption', 'upgrade', 'halving']
            is_important = any(keyword in title.lower() for keyword in important_keywords)

            for symbol in SYMBOLS:
                pattern = r'\b' + re.escape(symbol.replace("USDT", "").upper()) + r'\b'
                if re.search(pattern, title.upper()):
                    insert_news(symbol, title, source, published_at, content, sentiment_score)
                    print(f"✅ خبر ذخیره شد برای {symbol} - {title[:40]}... {'(مهم)' if is_important else ''}")

    except Exception as e:
        print(f"❌ خطا در دریافت اخبار: {e}")

if __name__ == "__main__":
    fetch_news()