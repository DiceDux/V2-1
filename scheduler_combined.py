import time
import subprocess

while True:
    print("📰 اجرای news_fetcher.py ...")
    subprocess.run(["python", "news_fetcher.py"])

    print("📈 اجرای market_index_updater.py ...")
    subprocess.run(["python", "market_index_updater.py"])

    print("⏳ صبر ۶۰ ثانیه...")
    time.sleep(60)
