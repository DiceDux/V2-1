# 🤖 DiceDux Trade

ربات هوشمند تحلیل‌گر و معامله‌گر ارز دیجیتال با پشتیبانی از چند ارز، تحلیل تکنیکال، الگوها، و چت هوشمند.

## 📌 امکانات

- تحلیل چندین ارز به‌صورت هم‌زمان (BTC, ETH, DOGE, ...)
- استفاده از اندیکاتورها: RSI, EMA, ATR, Volume, CH
- تشخیص الگوهای قیمتی (Double Bottom, Double Top)
- سیستم هوشمند تحلیل سیگنال خرید / فروش
- سیستم Backtesting برای بررسی عملکرد در گذشته
- موتور اجرای معاملات در حالت شبیه‌سازی
- رابط چت هوشمند برای گفتگو با ربات در مورد تحلیل‌ها

## 🛠️ اجرای پروژه

```bash
git clone <repo>
cd DiceDuxTrade
pip install -r requirements.txt
python main.py
🧠 مکالمه با ربات
python
Copy
Edit
from ai.chat_interface import chat_with_ai
print(chat_with_ai("الان چی بخرم؟"))
📂 ساختار پروژه
arduino
Copy
Edit
DiceDuxTrade/
├── main.py                    ← راه‌انداز اصلی پروژه
├── config.py                  ← تنظیمات کلی (مثل لیست ارزها، تایم‌فریم، API key)
├── main_ml.py
├── ai/
│   ├── ai_engine.py           ← مغز هوش مصنوعی (تحلیل‌گر)
│   └── chat_interface.py      ← چت با AI درباره تحلیل‌ها
├── data/
│   └── fetch_coinex.py        ← دریافت کندل‌ها از CoinEx
├── indicators/
│   ├── rsi.py
│   ├── ema.py
│   ├── atr.py
│   ├── volume.py
│   ├── ch.py
│   └── patterns.py            ← تشخیص الگوهای معروف
├── strategy/
│   └── decision_engine.py     ← تصمیم نهایی خرید/فروش
├── trading/
│   ├── simulation.py          ← اجرای مجازی معاملات
│   └── real_order.py          ← اتصال به CoinEx (برای بعد)
├── backtesting/
│   └── backtest.py            ← تست روی دیتای تاریخی
├── utils/
│   ├── logger.py
│   └── tools.py
├── models/
│   └── model.pkl              ← مدل یادگیری (قابل آموزش با دیتا)
├── notebooks/
│   ├── sample_candles.csv
│   └── train_ai_model.ipynb   ← دفترچه آموزش مدل هوش مصنوعی
├── requirements.txt
└── README.md
✅ وضعیت فعلی
✅ فقط حالت شبیه‌سازی
🛑 سفارشات واقعی غیرفعال (امنیت اولویت دارد)

Developed with ❤️ by DiceDux

yaml
Copy
Edit

---

✅ همه فایل‌ها آماده‌س.  
الآن کل پروژه رو برات توی ساختار `DiceDuxTrade` می‌ریزم، فایل zip می‌سازم و اینجا برات آپلود می‌کنم.

⏳ فقط چند ثانیه صبر کن...