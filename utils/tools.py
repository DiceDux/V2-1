# utils/tools.py

from datetime import datetime

def timestamp_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

def round_percent(value: float, decimals: int = 2) -> float:
    return round(value * 100, decimals)

def format_price(price: float) -> str:
    return f"{price:,.2f}"
