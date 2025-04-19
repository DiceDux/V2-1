import joblib
import pandas as pd
import os
from utils.logger import setup_logger
from data.data_manager import get_features_from_db
from feature_engineering_full_ultra_v2 import extract_features_full
from ai.fundamental_analyzer import get_fundamental_score_from_db

logger = setup_logger("AI", "ai_decisions.log")

# مسیر مطلق مدل
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ensemble_model_multi_with_fundamental.pkl'))

# بارگذاری مدل
model = joblib.load(MODEL_PATH)
label_map = {0: "sell", 1: "hold", 2: "buy"}  # اگر کلاس‌ها در مدل بر اساس عدد باشند

def predict_signal_from_model(df: pd.DataFrame, symbol=None, interval=None, verbose: bool = False) -> dict:
    # استخراج ویژگی‌ها
    features_df = extract_features_full(df)

    # افزودن امتیاز خبری
    news_score = get_fundamental_score_from_db(symbol, int(df["timestamp"].iloc[-1]))
    features_df["news_sentiment"] = news_score

    # انتخاب آخرین ردیف برای پیش‌بینی
    latest = features_df.iloc[-1:]

    # بررسی ویژگی‌های گمشده
    expected_features = model.feature_names_
    missing_features = [col for col in expected_features if col not in latest.columns]
    if missing_features:
        logger.warning(f"Missing features for {symbol}: {missing_features}")
        # پر کردن ویژگی‌های گمشده با مقدار پیش‌فرض (مثلاً 0)
        for feature in missing_features:
            latest[feature] = 0
        logger.info(f"Filled missing features for {symbol} with 0.")

    # تطبیق ویژگی‌ها با مدل
    latest = latest[expected_features]

    # پیش‌بینی
    try:
        prediction = model.predict(latest)[0]
        proba = model.predict_proba(latest)[0]
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {e}")
        return {
            "action": "hold",
            "confidence": 0.0,
            "features": latest.iloc[0].to_dict(),
            "catboost_confidence": 0.0,
            "lightgbm_confidence": 0.0,
            "xgboost_fundamental_confidence": 0.0
        }

    confidence = float(max(proba))
    try:
        predicted_class = label_map.get(int(prediction), "hold")
    except ValueError:
        predicted_class = str(prediction).lower()

    # رفع مشکل مربوط به confidence مدل‌ها
    # توجه: متغیرهای catboost_proba, lightgbm_proba, xgboost_proba تعریف نشده‌اند
    # فرض می‌کنیم که proba شامل احتمالات ترکیبی است (به دلیل استفاده از VotingClassifier)
    # اگر مدل‌های جداگانه در دسترس هستند، باید به روش دیگری این مقادیر را استخراج کنید
    result = {
        "action": predicted_class,
        "confidence": round(confidence, 3),
        "features": latest.iloc[0].to_dict(),
        "catboost_confidence": confidence,  # جایگزین موقت
        "lightgbm_confidence": confidence,  # جایگزین موقت
        "xgboost_fundamental_confidence": confidence  # جایگزین موقت
    }

    if verbose:
        logger.info("[%s] ویژگی‌ها:%s", symbol, latest.to_string(index=False))
        logger.info("[%s] تصمیم مدل: %s | اعتماد: %s", symbol, result['action'].upper(), result['confidence'])
        print("🧠 ویژگی‌های ورودی مدل:")
        print(latest)
        print(f"📢 تصمیم مدل: {result['action'].upper()} | اعتماد: {result['confidence']}")

    return result