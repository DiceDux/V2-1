import re
import pandas as pd

log_file_path = "logs/ai_decisions.log"

data = []
with open(log_file_path, encoding="utf-8") as f:
    lines = f.readlines()

    for i, line in enumerate(lines):
        # فقط خط تصمیم مدل
        if "تصمیم مدل:" in line:
            match = re.search(r"\[(.*?)\].*?تصمیم مدل: (\w+).*?اعتماد: ([\d\.]+)", line)
            if match:
                symbol = match.group(1)
                action = match.group(2).upper()
                confidence = float(match.group(3))

                # سعی کن خط قبلی که شامل news_sentiment است را پیدا کنی
                sentiment = None
                for j in range(i - 1, max(0, i - 10), -1):
                    if "news_sentiment" in lines[j]:
                        sentiment_match = re.search(r"news_sentiment\s+([-\d\.eE]+)", lines[j])
                        if sentiment_match:
                            sentiment = float(sentiment_match.group(1))
                            break

                data.append({
                    "symbol": symbol,
                    "action": action,
                    "confidence": confidence,
                    "news_sentiment": sentiment,
                })

# ساخت دیتافریم نهایی
df = pd.DataFrame(data)

if df.empty:
    print("⛔ هیچ داده‌ای قابل استخراج نیست.")
else:
    print("[+] آمار خلاصه news_sentiment بر اساس خروجی مدل:")
    print(df.groupby("action")["news_sentiment"].describe())

    import matplotlib.pyplot as plt

    df.boxplot(column="news_sentiment", by="action")
    plt.title("News Sentiment Distribution by Model Action")
    plt.suptitle("")
    plt.ylabel("news_sentiment")
    plt.xlabel("Model Action")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
