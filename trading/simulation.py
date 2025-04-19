# trading/simulation.py

from data.data_manager import (
    get_wallet_balance, update_wallet_balance,
    get_position, insert_position, delete_position,
    save_trade_record
)

TRADE_PERCENT = 0.2

def simulate_trade(symbol, action, price, confidence=0.0):
    print(f"⚙️ [simulate] شروع ترید | symbol={symbol}, action={action}, price={price}")

    balance = float(get_wallet_balance())

    if action == "buy":
        trade_amount = balance * TRADE_PERCENT
        if trade_amount > balance:
            print("⚠️ موجودی کافی برای خرید وجود ندارد.")
            return {"balance": balance}
        
        quantity = trade_amount / price
        tp_price = price * 1.10  # TP اولیه = 10٪ بالاتر
        sl_price = price * 0.90  # SL اولیه = 10٪ پایین‌تر
        update_wallet_balance(balance - trade_amount)
        insert_position(symbol, "buy", price, quantity, tp_price, sl_price, tp_step=1, last_price=price)
        return {"balance": balance - trade_amount}

    elif action == "sell":
        position = get_position(symbol)
        if not position:
            print("⚠️ هیچ پوزیشنی برای فروش پیدا نشد.")
            return {
                "balance": balance,
                "quantity": 0,
                "exit_price": price,
                "profit_percent": 0
            }

        entry_price = float(position["entry_price"])
        quantity = float(position["quantity"])
        sell_amount = quantity * price
        profit_percent = ((price - entry_price) / entry_price) * 100

        update_wallet_balance(balance + sell_amount)
        delete_position(symbol)

        return {
            "balance": balance + sell_amount,
            "quantity": quantity,
            "exit_price": price,
            "profit_percent": profit_percent
        }

