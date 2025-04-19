# backtesting/backtest.py

from data.fetch_coinex import fetch_candles
from ai.ai_engine import analyze_market
from config import SYMBOLS, INITIAL_BALANCE

def backtest(symbol: str) -> dict:
    """
    بک‌تست استراتژی روی یک ارز
    :param symbol: مانند BTCUSDT
    :return: دیکشنری شامل نتایج تست
    """
    df = fetch_candles(symbol)
    balance = INITIAL_BALANCE
    position = None
    entry_price = 0
    trades = []

    for i in range(30, len(df)):  # از کندل 30 به بعد چون اندیکاتورها نیاز به دیتای قبلی دارن
        sample = df.iloc[:i+1]
        signal = analyze_market(sample)

        if signal["action"] == "buy" and position is None:
            position = "long"
            entry_price = sample["close"].iloc[-1]
            trades.append({"type": "buy", "price": entry_price})

        elif signal["action"] == "sell" and position == "long":
            exit_price = sample["close"].iloc[-1]
            profit = (exit_price - entry_price) / entry_price
            balance *= (1 + profit)
            trades.append({"type": "sell", "price": exit_price, "profit": round(profit * 100, 2)})
            position = None

    return {
        "symbol": symbol,
        "final_balance": round(balance, 2),
        "trades": trades,
        "total_trades": len(trades) // 2,
        "gain_percent": round((balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100, 2)
    }
