import requests
import time
from datetime import datetime, timedelta
from ai.news_sentiment_ai import analyze_sentiment
from data.data_manager import insert_news
import pytz

API_KEY = "da4aff9ddef8f6534f149d3803e2e0420958c5f4"

SYMBOLS = ["XAUTUSDT", "PESTOUSDT",
           "SOLUSDT", "EOSUSDT", "QNTUSDT", "FOXUSDT", "RSS3USDT", "SUSDT", "API3USDT",
           "LINKUSDT", "THETAUSDT"]

def fetch_news(target_news_per_symbol=100, days_back=30):
    print("📰 در حال دریافت اخبار...")
    # تاریخ شروع (30 روز قبل) با منطقه زمانی UTC
    start_date = datetime.now(pytz.UTC) - timedelta(days=days_back)
    print(f"📅 جمع‌آوری اخبار از {start_date} به بعد...")

    for symbol in SYMBOLS:
        print(f"🔍 بررسی اخبار برای {symbol}...")
        # به جای حذف "USDT"، از نماد اصلی (مثل BTCUSDT) استفاده می‌کنیم
        base_symbol = symbol.replace("USDT", "")  # فقط برای جست‌وجو در عنوان خبر
        inserted = 0
        page = 1

        while inserted < target_news_per_symbol:
            try:
                url = f"https://cryptopanic.com/api/v1/posts/?auth_token={API_KEY}&public=true&page={page}"
                res = requests.get(url, timeout=10)
                if res.status_code == 429:
                    retry_after = int(res.headers.get("Retry-After", 60))
                    print(f"⚠️ Rate limit. صبر می‌کنیم {retry_after} ثانیه...")
                    time.sleep(retry_after)
                    continue
                elif res.status_code != 200:
                    print(f"⚠️ خطا در دریافت اخبار ({res.status_code}) برای {symbol}")
                    break

                data = res.json()
                posts = data.get("results", [])
                if not posts:
                    print(f"❌ خبری برای {symbol} یافت نشد.")
                    break

                for post in posts:
                    if inserted >= target_news_per_symbol:
                        break

                    # بررسی تاریخ خبر
                    published_at = post.get("published_at", datetime.now(pytz.UTC).isoformat())
                    try:
                        published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S%z")
                    except ValueError as e:
                        print(f"⚠️ خطا در پارس تاریخ {published_at}: {e}")
                        continue

                    if published_date < start_date:
                        print(f"⏳ اخبار قدیمی‌تر از {days_back} روز برای {symbol}. توقف...")
                        break

                    title = post.get("title", "")
                    content = post.get("body") or title
                    # فیلتر آزادتر: اخبار مرتبط با کریپتو یا نماد (بدون USDT در جست‌وجو)
                    if not (base_symbol.lower() in title.lower() or "crypto" in title.lower() or "bitcoin" in title.lower()):
                        continue

                    score = analyze_sentiment(content)
                    source = post.get("source", {}).get("title", "unknown")

                    try:
                        insert_news(
                            symbol=symbol,  # اینجا از نماد اصلی (مثل BTCUSDT) استفاده می‌کنیم
                            title=title,
                            source=source,
                            published_at=published_at,
                            content=content,
                            sentiment_score=score
                        )
                        inserted += 1
                        print(f"✅ خبر {inserted} برای {symbol} ذخیره شد. امتیاز احساسات: {score:.3f}")
                        time.sleep(0.2)  # تأخیر برای جلوگیری از Rate Limit
                    except Exception as e:
                        print("❌ خطا در درج خبر:", e)

                page += 1
                time.sleep(1)  # تأخیر بین درخواست‌ها برای رعایت Rate Limit (5 req/s)

            except Exception as e:
                print(f"❌ خطا در دریافت خبر: {e}")
                break

        print(f"📊 جمع‌آوری اخبار برای {symbol} به پایان رسید. تعداد اخبار: {inserted}")

if __name__ == "__main__":
    fetch_news(target_news_per_symbol=100, days_back=30)