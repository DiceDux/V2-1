# trading/trade_status.py

import json
import os

STATUS_FILE = "trade_status.json"

def get_trade_status():
    if not os.path.exists(STATUS_FILE):
        return "running"
    try:
        with open(STATUS_FILE, "r") as f:
            data = json.load(f)
            return data.get("status", "running")
    except json.JSONDecodeError:
        # فایل خالی یا خراب
        return "running"

def set_trade_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump({"status": status}, f)
