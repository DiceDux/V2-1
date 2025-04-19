# trading/real_order.py

from data.data_manager import save_trade_record, save_balance

def execute_real_order(symbol, action, price, balance):
    print(f"⚙️ [real] اجرای سفارش واقعی | symbol={symbol}, action={action}, price={price}")

    # در اینجا می‌تونی به CoinEx API یا صرافی دیگر وصل بشی
    # برای این نسخه اولیه، فرض می‌گیریم فقط ثبت در دیتابیس انجام می‌شه

    save_trade_record(symbol, action, price, balance)
    save_balance(symbol, balance)

    print(f"✅ سفارش واقعی ثبت شد | موجودی: {balance}")
    return {"balance": balance}
