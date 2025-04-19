# test_predict.py
from ai.ai_model_runner import predict_signal_from_model
from data.data_manager import get_candle_data

symbol = "BTCUSDT"
df = get_candle_data(symbol)
if not df.empty:
    signal = predict_signal_from_model(df, symbol=symbol, interval="5min", verbose=True)
    print(signal)
else:
    print("داده‌های کندل خالیه!")