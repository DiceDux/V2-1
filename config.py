# config.py

# 🔧 تنظیمات کلی ربات DiceDux Trade

# حالت اجرا: "simulation" یا "real"
TRADE_MODE = "simulation"

# لیست ارزهایی که ربات روی آن‌ها کار می‌کند
SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "XUSDT", "FARTCOINUSDT", "PIPPINUSDT", "XAUTUSDT", "PESTOUSDT",
           "SOLUSDT", "EOSUSDT", "QNTUSDT", "FOXUSDT", "RSS3USDT", "SUSDT", "API3USDT",
            "LINKUSDT", "THETAUSDT",]

# تایم‌فریم تحلیل (دقیقه‌ای)
TIMEFRAME_MINUTES = 240  # یعنی 4 ساعته

# تنظیمات CoinEx API
COINEX_BASE_URL = "https://api.coinex.com"
COINEX_KLINE_ENDPOINT = "/market/kline"
COINEX_API_KEY = "F5498FB9C0A34C23B3FE704FE174105C"
COINEX_API_SECRET = "AA4173A9CE1E147E9B8725C2F8E3D987D95ED8AF095ECD16"

# مقدار سرمایه مجازی در حالت شبیه‌سازی (هر ارز)
INITIAL_BALANCE = 1000

# تعداد کندل گذشته برای تحلیل
CANDLE_HISTORY_LIMIT = 200

# config.py
MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",  # ← اگه رمز داری اینجا واردش کن
    "database": "dicedux_db"
}
