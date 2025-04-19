import joblib
import pandas as pd
import os
import numpy as np
from utils.logger import setup_logger
from data.data_manager import get_features_from_db
from feature_engineering_full_ultra_v2 import extract_features_full

logger = setup_logger("AI", "ai_decisions.log")

# مسیر مدل‌های جدید
CATBOOST_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_model.pkl'))
LIGHTGBM_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'lightgbm_model.pkl'))
XGBOOST_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl'))

# مسیر مدل‌های قدیمی
ENSEMBLE_OLD_1_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble_model_multi_with_fundamental_oldold.pkl'))
ENSEMBLE_OLD_2_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble_model_multi_with_fundamental_old.pkl'))
CATBOOST_OLD_STANDALONE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'catboost_model_multi_old.cbm'))

# لود کردن مدل‌های جدید
catboost_model = joblib.load(CATBOOST_MODEL_PATH)
lightgbm_model = joblib.load(LIGHTGBM_MODEL_PATH)
xgboost_model = joblib.load(XGBOOST_MODEL_PATH)

# لود کردن مدل‌های قدیمی (با مدیریت خطا)
try:
    ensemble_old_1 = joblib.load(ENSEMBLE_OLD_1_PATH)
except FileNotFoundError:
    ensemble_old_1 = None
    logger.warning("مدل قدیمی Ensemble 1 پیدا نشد.")

try:
    ensemble_old_2 = joblib.load(ENSEMBLE_OLD_2_PATH)
except FileNotFoundError:
    ensemble_old_2 = None
    logger.warning("مدل قدیمی Ensemble 2 پیدا نشد.")

try:
    catboost_old_standalone = joblib.load(CATBOOST_OLD_STANDALONE_PATH)
except FileNotFoundError:
    catboost_old_standalone = None
    logger.warning("مدل قدیمی CatBoost Standalone پیدا نشد.")

label_map = {0: "sell", 1: "hold", 2: "buy"}

def predict_signal_from_model(df: pd.DataFrame, symbol=None, interval=None, verbose: bool = False) -> dict:
    print(f"ورودی تابع: symbol={symbol}, df_shape={df.shape}")
    
    df['symbol'] = symbol
    print(f"ستون‌های df بعد از اضافه کردن symbol: {df.columns.tolist()}")

    try:
        features_df = extract_features_full(df)
        print(f"features_df_shape: {features_df.shape}")
        print(f"features_df_columns: {features_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error in extract_features_full for {symbol}: {e}")
        print(f"❌ خطا در extract_features_full برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    try:
        latest = features_df.iloc[-1:]
        print(f"latest_shape: {latest.shape}")
        print(f"latest_columns: {latest.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error selecting latest row for {symbol}: {e}")
        print(f"❌ خطا در انتخاب آخرین ردیف برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    try:
        expected_features = [
            'ema20', 'ema50', 'ema200', 'rsi', 'atr', 'tema20', 'dema20', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_mid', 'bb_width', 'bb_squeeze', 'keltner_upper', 'keltner_lower',
            'donchian_upper', 'donchian_lower', 'obv', 'vwap', 'adx', 'breakout', 'breakdown', 'volume_spike',
            'vwap_buy_signal', 'rsi_slope', 'macd_slope', 'rsi_macd_converge', 'stoch_rsi', 'cci', 'willr',
            'mfi', 'roc', 'momentum', 'psar', 'ult_osc', 'ichimoku_a', 'ichimoku_b', 'ichimoku_base',
            'ichimoku_conv', 'daily_return', 'candle_length_pct', 'candle_length', 'candle_body', 'upper_wick',
            'lower_wick', 'body_to_range', 'relative_volume', 'ema_cross', 'trend_strength', 'trend_age',
            'range_pct', 'range_spike', 'ha_close', 'ha_open', 'gap_up', 'gap_down', 'ema_spread',
            'ema_compression', 'bullish_candles', 'bullish_streak', 'avg_true_body', 'div_rsi', 'div_macd',
            'div_obv', 'confirmed_rsi_div', 'volatility_14', 'z_score', 'doji', 'hammer', 'inv_hammer',
            'hanging_man', 'engulfing_bull', 'engulfing_bear', 'morning_star', 'evening_star', 'harami_bull',
            'harami_bear', 'piercing_line', 'three_white_soldiers', 'spinning_top', 'marubozu', 'three_black_crows',
            'combo_signal', 'hour', 'session_asia', 'session_europe', 'session_us', 'double_top', 'head_shoulders',
            'support_zone', 'resistance_zone', 'support_bounce', 'resistance_reject', 'cup', 'handle',
            'cup_and_handle', 'higher_highs', 'lower_lows', 'diamond_top', 'flag_pole', 'flag_body', 'flag_pattern',
            'fibo_0_5_bounce', 'fibo_0_618_bounce', 'spx_index', 'dxy_index', 'btc_d', 'usdt_d', 'spx', 'dxy',
            'low_volatility', 'volume_mean', 'low_volume', 'weak_trend', 'low_adx', 'low_z_score', 'chaikin_osc',
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'vwap_new', 'obv_new',
            'volume_score', 'btc_dominance', 'usdt_dominance',
            'day_of_week', 'month', 'price_to_30d_mean'
        ] + [f"news_emb_{i}" for i in range(768)]
        print(f"expected_features: {expected_features}")
    except Exception as e:
        logger.error(f"Error determining expected features for {symbol}: {e}")
        print(f"❌ خطا در تعیین ویژگی‌های مورد انتظار برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": latest.iloc[0].to_dict() if not latest.empty else {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    try:
        missing_features = [col for col in expected_features if col not in latest.columns]
        if missing_features:
            logger.warning(f"Missing features for {symbol}: {missing_features}")
            print(f"⚠️ ویژگی‌های گمشده برای {symbol}: {missing_features}")
            for feature in missing_features:
                latest[feature] = 0
            logger.info(f"Filled missing features for {symbol} with 0.")
    except Exception as e:
        logger.error(f"Error checking missing features for {symbol}: {e}")
        print(f"❌ خطا در بررسی ویژگی‌های گمشده برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": latest.iloc[0].to_dict() if not latest.empty else {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    try:
        latest = latest[expected_features]
        print(f"ویژگی‌های نهایی: {latest.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error aligning features for {symbol}: {e}")
        print(f"❌ خطا در تطبیق ویژگی‌ها برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": latest.iloc[0].to_dict() if not latest.empty else {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    # پیش‌بینی با هر مدل
    try:
        # احتمالات مدل‌های جدید
        catboost_proba = catboost_model.predict_proba(latest)[0]
        lightgbm_proba = lightgbm_model.predict_proba(latest)[0]
        xgboost_proba = xgboost_model.predict_proba(latest)[0]
        
        # احتمالات مدل‌های قدیمی (اگه وجود داشته باشن)
        ensemble_old_1_proba = ensemble_old_1.predict_proba(latest)[0] if ensemble_old_1 else np.zeros(3)
        ensemble_old_2_proba = ensemble_old_2.predict_proba(latest)[0] if ensemble_old_2 else np.zeros(3)
        catboost_old_standalone_proba = catboost_old_standalone.predict_proba(latest)[0] if catboost_old_standalone else np.zeros(3)
        
        # ترکیب احتمالات با وزن
        weights = [0.25, 0.25, 0.20, 0.10, 0.10, 0.10]  # وزن‌ها برای CatBoost جدید، LightGBM جدید، XGBoost جدید، Ensemble قدیمی 1، Ensemble قدیمی 2، CatBoost قدیمی
        combined_proba = (
            weights[0] * catboost_proba +
            weights[1] * lightgbm_proba +
            weights[2] * xgboost_proba +
            weights[3] * ensemble_old_1_proba +
            weights[4] * ensemble_old_2_proba +
            weights[5] * catboost_old_standalone_proba
        ) / sum(weights)
        
        prediction = np.argmax(combined_proba)
        confidence = float(max(combined_proba))
        print(f"combined_proba: {combined_proba}, prediction: {prediction}, confidence: {confidence}")
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        print(f"❌ خطا در پیش‌بینی برای {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": latest.iloc[0].to_dict() if not latest.empty else {},
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_confidence": 0.0
        }

    try:
        predicted_class = label_map.get(int(prediction), "hold")
    except ValueError:
        predicted_class = str(prediction).lower()

    result = {
        "action": predicted_class,
        "confidence": round(confidence, 3),
        "features": latest.iloc[0].to_dict(),
        "catboost_confidence": round(float(max(catboost_proba)), 3),
        "lightgbm_confidence": round(float(max(lightgbm_proba)), 3),
        "xgboost_confidence": round(float(max(xgboost_proba)), 3)
    }

    if verbose:
        logger.info("[%s] ویژگی‌ها:%s", symbol, latest.to_string(index=False))
        logger.info("[%s] تصمیم مدل: %s | اعتماد: %s", symbol, result['action'].upper(), result['confidence'])
        print("🧠 ویژگی‌های ورودی مدل:")
        print(latest)
        print(f"📢 تصمیم مدل: {result['action'].upper()} | اعتماد: {result['confidence']}")

    return result