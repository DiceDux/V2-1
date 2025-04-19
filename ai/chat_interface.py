# ai/chat_interface.py

from ai.ai_engine import analyze_market
from data.fetch_coinex import fetch_candles
from config import SYMBOLS

def chat_with_ai(question: str) -> str:
    """
    پاسخ ساده به سوالات مرتبط با تحلیل ارزها
    """
    question = question.lower()

    if "کدوم ارز" in question or "سیگنال خرید" in question:
        suggestions = []
        for symbol in SYMBOLS:
            df = fetch_candles(symbol)
            signal = analyze_market(df)
            if signal["action"] == "buy":
                suggestions.append(f"{symbol} ✅ (اعتماد: {signal['confidence']})")
        if suggestions:
            return "🔍 ارزهای مناسب برای خرید:\n" + "\n".join(suggestions)
        else:
            return "فعلاً سیگنال خریدی شناسایی نشده."

    elif "تحلیل" in question or "وضعیت" in question:
        responses = []
        for symbol in SYMBOLS:
            df = fetch_candles(symbol)
            signal = analyze_market(df)
            responses.append(f"{symbol}: {signal['action'].upper()} (اعتماد: {signal['confidence']})")
        return "📊 تحلیل ارزها:\n" + "\n".join(responses)

    elif "سود" in question or "ضرر" in question:
        return "📈 سود/ضرر واقعی بعد از اجرای معاملات شبیه‌سازی‌شده محاسبه خواهد شد."

    else:
        return "❓ سوالت مرتبط با ترید نبود، لطفاً درباره تحلیل، خرید، فروش یا سود/ضرر بپرس."
