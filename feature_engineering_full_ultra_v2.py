import pandas as pd
import numpy as np
import ta
from data.data_manager import get_recent_news_texts
from data.data_manager import get_index_value
from ai.news_sentiment_ai import analyze_sentiment
from data.market_index_manager import get_index
from ai.fundamental_analyzer import analyze_fundamentals

def calculate_ichimoku_cloud(df):
    """محاسبه اندیکاتور Ichimoku Cloud"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou_span = close.shift(-26)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }

def calculate_vwap(df):
    """محاسبه VWAP"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_obv(df):
    """محاسبه OBV"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def calculate_chaikin_oscillator(df: pd.DataFrame) -> pd.Series:
    """
    محاسبه Chaikin Oscillator به صورت دستی
    """
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_multiplier = money_flow_multiplier.fillna(0)
    money_flow_volume = money_flow_multiplier * df['volume']
    ad = money_flow_volume.cumsum()
    short_ema = ad.ewm(span=3, adjust=False).mean()
    long_ema = ad.ewm(span=10, adjust=False).mean()
    chaikin_osc = short_ema - long_ema
    return chaikin_osc

def extract_features_full(df: pd.DataFrame, df_btc=None, df_eth=None, df_doge=None) -> pd.DataFrame:
    df = df.copy()
    features = {}   

    # Basic Indicators
    features['ema20'] = df['close'].ewm(span=20).mean()
    features['ema50'] = df['close'].ewm(span=50).mean()
    features['ema200'] = df['close'].ewm(span=200).mean()
    features['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    features['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    features['tema20'] = df['close'].ewm(span=20).mean().ewm(span=20).mean().ewm(span=20).mean()
    features['dema20'] = 2 * df['close'].ewm(span=20).mean() - df['close'].ewm(span=20).mean().ewm(span=20).mean()
    macd = ta.trend.MACD(close=df['close'])
    features['macd'] = macd.macd()
    features['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    features['bb_upper'] = bb.bollinger_hband()
    features['bb_lower'] = bb.bollinger_lband()
    features['bb_mid'] = bb.bollinger_mavg()
    features['bb_width'] = features['bb_upper'] - features['bb_lower']
    features['bb_squeeze'] = features['bb_width'] < features['bb_width'].rolling(window=20).mean() * 0.8
    features['keltner_upper'] = features['ema20'] + 2 * features['atr']
    features['keltner_lower'] = features['ema20'] - 2 * features['atr']
    features['donchian_upper'] = df['high'].rolling(window=20).max()
    features['donchian_lower'] = df['low'].rolling(window=20).min()
    features['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    features['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    features['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
    features['breakout'] = df['close'] > features['donchian_upper']
    features['breakdown'] = df['close'] < features['donchian_lower']
    features['volume_spike'] = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    features['vwap_buy_signal'] = (df['close'] > features['vwap']) & (features['volume_spike'])
    features['rsi_slope'] = features['rsi'].diff()
    features['macd_slope'] = features['macd'].diff()
    features['rsi_macd_converge'] = (features['rsi_slope'] > 0) & (features['macd_slope'] > 0)
    features['stoch_rsi'] = ta.momentum.StochRSIIndicator(close=df['close']).stochrsi()
    features['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    features['willr'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
    features['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
    features['roc'] = ta.momentum.ROCIndicator(close=df['close'], window=10).roc()
    features['momentum'] = df['close'] - df['close'].shift(10)
    features['psar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    bp = df['close'] - pd.concat([df['low'], df['close'].shift(1)], axis=1).min(axis=1)
    tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    features['ult_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
    ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    features['ichimoku_a'] = ichimoku.ichimoku_a()
    features['ichimoku_b'] = ichimoku.ichimoku_b()
    features['ichimoku_base'] = ichimoku.ichimoku_base_line()
    features['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    features['daily_return'] = df['close'].pct_change()
    features["candle_length_pct"] = (df["high"] - df["low"]) / df["open"]
    features['candle_length'] = df['high'] - df['low']
    features['candle_body'] = abs(df['close'] - df['open'])
    features['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    features['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    features['body_to_range'] = features['candle_body'] / features['candle_length'].replace(0, 1)
    features['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
    features['ema_cross'] = (features['ema20'] > features['ema50']).astype(int)
    features['trend_strength'] = (features['ema20'] - features['ema50']) / features['ema50']
    features['trend_age'] = features['ema_cross'].groupby((features['ema_cross'] != features['ema_cross'].shift()).cumsum()).cumcount() + 1
    features['range_pct'] = features['candle_length'] / features['candle_length'].rolling(20).mean()
    features['range_spike'] = features['range_pct'] > 1.5
    features['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    features['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    features['gap_up'] = df['open'] > df['close'].shift(1) * 1.01
    features['gap_down'] = df['open'] < df['close'].shift(1) * 0.99
    features['ema_spread'] = (features['ema200'] - features['ema20']) / features['ema20']
    features['ema_compression'] = features['ema_spread'].abs() < features['ema_spread'].rolling(20).std()
    features['bullish_candles'] = (df['close'] > df['open']).astype(int)
    features['bullish_streak'] = features['bullish_candles'].groupby((features['bullish_candles'] != features['bullish_candles'].shift()).cumsum()).cumcount() + 1
    features['avg_true_body'] = features['candle_body'].rolling(window=14).mean()
    features['div_rsi'] = df['close'].diff() - features['rsi'].diff()
    features['div_macd'] = df['close'].diff() - features['macd'].diff()
    features['div_obv'] = df['close'].diff() - features['obv'].diff()
    features['confirmed_rsi_div'] = (features['div_rsi'] > 0) & (features['adx'] < 25)
    features['volatility_14'] = df['close'].rolling(window=14).std()
    features['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    features['doji'] = abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1
    features['hammer'] = ((df['high'] - df['close']) < 2 * (df['close'] - df['low'])) & (df['close'] > df['open'])
    features['inv_hammer'] = ((df['close'] - df['low']) < 2 * (df['high'] - df['close'])) & (df['close'] > df['open'])
    features['hanging_man'] = ((df['high'] - df['close']) < 2 * (df['close'] - df['low'])) & (df['close'] < df['open'])
    features['engulfing_bull'] = (df['close'] > df['open']) & (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1))
    features['engulfing_bear'] = (df['close'] < df['open']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    features['morning_star'] = (df['close'].shift(2) > df['open'].shift(2)) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    features['evening_star'] = (df['close'].shift(2) < df['open'].shift(2)) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    features['harami_bull'] = (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    features['harami_bear'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    features['piercing_line'] = (df['open'] < df['close'].shift(1)) & (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)
    features['three_white_soldiers'] = (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2))
    features['spinning_top'] = features['body_to_range'].between(0.2, 0.5)
    features['marubozu'] = features['body_to_range'] > 0.9
    features['three_black_crows'] = (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2))
    features['combo_signal'] = (features['engulfing_bull']) & (features['rsi'] < 30) & (features['breakout'])
    features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    features['session_asia'] = features['hour'].between(1, 9)
    features['session_europe'] = features['hour'].between(9, 17)
    features['session_us'] = features['hour'].between(17, 24)
    features['double_top'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    features['head_shoulders'] = (df['high'].shift(2) < df['high'].shift(1)) & (df['high'].shift(1) > df['high']) & (df['high'].shift(2) < df['high'])
    features['support_zone'] = df['low'].rolling(20).min()
    features['resistance_zone'] = df['high'].rolling(20).max()
    features['support_bounce'] = (df['low'] <= features['support_zone'] * 1.01)
    features['resistance_reject'] = (df['high'] >= features['resistance_zone'] * 0.99)
    features['cup'] = (df['close'].rolling(window=50).mean() < df['close'].rolling(window=100).mean()) & (df['close'] > df['close'].rolling(window=50).mean())
    features['handle'] = (features['cup']) & (df['close'] < df['close'].shift(1))
    features['cup_and_handle'] = features['handle'].rolling(window=5).sum() > 0
    features['higher_highs'] = df['high'] > df['high'].shift(1)
    features['lower_lows'] = df['low'] < df['low'].shift(1)
    features['diamond_top'] = (features['higher_highs'].rolling(window=5).sum() >= 3) & (features['lower_lows'].rolling(window=5).sum() >= 3)
    features['flag_pole'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
    features['flag_body'] = (df['close'] < df['close'].shift(1)) & (df['close'] > df['close'].shift(2))
    features['flag_pattern'] = (features['flag_pole'].rolling(window=3).sum() == 1) & (features['flag_body'].rolling(window=3).sum() == 2)
    swing_high = df['high'].rolling(window=50).max()
    swing_low = df['low'].rolling(window=50).min()
    fibo_0_5 = swing_low + (swing_high - swing_low) * 0.5
    fibo_0_618 = swing_low + (swing_high - swing_low) * 0.618
    features['fibo_0_5_bounce'] = abs(df['close'] - fibo_0_5) / fibo_0_5 < 0.01
    features['fibo_0_618_bounce'] = abs(df['close'] - fibo_0_618) / fibo_0_618 < 0.01
    features['spx_index'] = df['timestamp'].apply(lambda ts: get_index('SPX', ts))
    features['dxy_index'] = df['timestamp'].apply(lambda ts: get_index('DXY', ts))
    features['btc_d'] = df['timestamp'].apply(lambda ts: get_index("BTC.D", ts))
    features['usdt_d'] = df['timestamp'].apply(lambda ts: get_index("USDT.D", ts))
    features['spx'] = df['timestamp'].apply(lambda ts: get_index("SPX", ts))
    features['dxy'] = df['timestamp'].apply(lambda ts: get_index("DXY", ts))
    features["low_volatility"] = features["volatility_14"] < features["volatility_14"].rolling(50).mean() * 0.8
    features["volume_mean"] = df["volume"].rolling(50).mean()
    features["low_volume"] = df["volume"] < features["volume_mean"]
    features["weak_trend"] = features["ema_spread"].abs() < features["ema_spread"].rolling(50).std()
    features["low_adx"] = features["adx"] < 20
    features["low_z_score"] = features["z_score"].abs() < 0.5
    features['chaikin_osc'] = calculate_chaikin_oscillator(df)

    # اضافه کردن Chaikin Oscillator
    features['chaikin_osc'] = calculate_chaikin_oscillator(df)

    # اندیکاتورهای جدید
    ichimoku = calculate_ichimoku_cloud(df)
    for key, value in ichimoku.items():
        features[key] = value

    features['vwap_new'] = calculate_vwap(df)
    features['obv_new'] = calculate_obv(df)

    # تحلیل فاندامنتال
    df_fund = analyze_fundamentals(df)
    # فقط فیچرهایی که نیاز داریم رو اضافه می‌کنیم
    if 'fundamental_score' not in features:
        features['fundamental_score'] = df_fund.get('fundamental_score', pd.Series(0, index=df.index))
    if 'news_score' not in features:
        features['news_score'] = df_fund.get('news_score', pd.Series(0, index=df.index))
    if 'volume_score' not in features:
        features['volume_score'] = df_fund.get('volume_score', pd.Series(0, index=df.index))

    # تحلیل اخبار
    if 'symbol' in df.columns and 'timestamp' in df.columns:
        sentiments = []
        for i, row in df.iterrows():
            try:
                ts = int(pd.to_datetime(row['timestamp']).timestamp())
                text = get_recent_news_texts(row['symbol'], ts)
                score = analyze_sentiment(text)
                sentiments.append(score)
            except Exception as e:
                print(f"⚠️ خطا در تحلیل خبر {row['symbol']} - {row['timestamp']}: {e}")
                sentiments.append(0.0)
        features['news_sentiment'] = pd.Series(sentiments, index=df.index)

    if 'timestamp' in df.columns:
        features['btc_dominance'] = df['timestamp'].apply(lambda ts: get_index('BTC.D', ts))
        features['usdt_dominance'] = df['timestamp'].apply(lambda ts: get_index('USDT.D', ts))
    
    for key in features:
        features[key] = features[key].reindex(df.index)

    features_df = pd.concat(features, axis=1)
    features_df = features_df.fillna(0)
    
    # چک کردن فیچرهای تکراری
    if features_df.columns.duplicated().any():
        print(f"⚠️ ستون‌های تکراری یافت شدند: {features_df.columns[features_df.columns.duplicated()].tolist()}")
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]
    
    result_df = pd.concat([df, features_df], axis=1)
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]  # حذف ستون‌های تکراری
    result_df = result_df.reset_index(drop=True)
    return result_df