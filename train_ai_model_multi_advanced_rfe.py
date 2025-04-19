import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.ensemble,看看看 VotingClassifier
import joblib
from data.data_manager import get_candle_data
from feature_engineering_full_ultra_v2 import extract_features_full
from config import SYMBOLS

# 🎯 تنظیمات
TIMEFRAME = '4h'
WINDOW_SIZE = 100
LABEL_LOOKAHEAD = 12

def label_data(df: pd.DataFrame) -> pd.DataFrame:
    df['future_return'] = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1
    df['label'] = 'Hold'
    df.loc[df['future_return'] > 0.005, 'label'] = 'Buy'
    df.loc[df['future_return'] < -0.005, 'label'] = 'Sell'
    return df.dropna()

def main():
    all_data = []

    for symbol in SYMBOLS:
        print(f"📊 دریافت داده برای {symbol}")
        try:
            df = get_candle_data(symbol=symbol, limit=1000)
            if df.empty:
                continue
        except Exception as e:
            print(f"❌ خطا در دریافت کندل برای {symbol}: {e}")
            continue

        df['symbol'] = symbol
        df_feat = extract_features_full(df)
        df_labeled = label_data(df_feat)
        df_labeled = df_labeled.copy()
        df_labeled['symbol'] = symbol
        all_data.append(df_labeled)

    if not all_data:
        print("⛔ هیچ داده‌ای برای آموزش پیدا نشد.")
        return

    df_final = pd.concat(all_data)
    df_final.dropna(inplace=True)
    if 'news_sentiment' not in df_final.columns:
        df_final['news_sentiment'] = 0.0

    total = len(df_final)
    non_null = df_final['news_sentiment'].notna().sum()
    percentage = (non_null / total) * 100
    print(f"📊 درصد داده‌هایی که news_sentiment دارند: {percentage:.2f}%  ({non_null}/{total})")

    # ویژگی‌های فاندامنتال
    fundamental_features = ['news_sentiment', 'fundamental_score', 'volume_score', 'news_score']
    # ویژگی‌های تکنیکال (همه ویژگی‌ها منهای فاندامنتال)
    technical_features = [col for col in df_final.columns if col not in fundamental_features + ['label', 'future_return', 'symbol']]

    # داده‌های کامل برای CatBoost و LightGBM
    X = df_final.drop(columns=['label', 'future_return', 'symbol'])
    y = df_final['label']

    # داده‌های فاندامنتال برای XGBoost
    df_fundamental = df_final[df_final['news_sentiment'] != 0]  # فقط داده‌هایی که news_sentiment دارن
    X_fundamental = df_fundamental[fundamental_features]
    y_fundamental = df_fundamental['label']

    print("🔍 ویژگی‌های تکنیکال:", technical_features)
    print("🔍 ویژگی‌های فاندامنتال:", fundamental_features)

    # تبدیل لیبل‌ها به عدد
    label_mapping = {'Sell': 0, 'Hold': 1, 'Buy': 2}
    y = y.map(label_mapping)
    y_fundamental = y_fundamental.map(label_mapping)

    # تقسیم داده‌ها برای CatBoost و LightGBM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # تقسیم داده‌ها برای XGBoost
    if not X_fundamental.empty:
        X_train_fund, X_test_fund, y_train_fund, y_test_fund = train_test_split(X_fundamental, y_fundamental, test_size=0.2, random_state=42)
    else:
        print("⚠️ داده کافی برای آموزش مدل فاندامنتال وجود ندارد.")
        return

    # بهینه‌سازی مدل برای RFE (CatBoost)
    model_for_rfe = CatBoostClassifier(verbose=0)
    rfe_params = {
        'iterations': [50, 100],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6]
    }
    rfe_search = GridSearchCV(model_for_rfe, rfe_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    rfe_search.fit(X_train, y_train)
    best_rfe_model = rfe_search.best_estimator_
    print("📊 بهترین پارامترهای مدل RFE:", rfe_search.best_params_)

    # استفاده از RFECV برای انتخاب ویژگی‌ها
    rfecv = RFECV(estimator=best_rfe_model, step=1, cv=3, scoring='f1_weighted')
    rfecv.fit(X_train, y_train)
    selected_features_rfe = X_train.columns[rfecv.support_]
    print("📊 ویژگی‌های منتخب RFE برای CatBoost و LightGBM:", selected_features_rfe.tolist())

    # انتخاب ویژگی برای XGBoost
    rfecv_fund = RFECV(estimator=best_rfe_model, step=1, cv=3, scoring='f1_weighted')
    rfecv_fund.fit(X_train_fund, y_train_fund)
    selected_fund_features_rfe = X_train_fund.columns[rfecv_fund.support_]
    print("📊 ویژگی‌های منتخب RFE برای XGBoost (فاندامنتال):", selected_fund_features_rfe.tolist())

    # بهینه‌سازی CatBoost
    catboost = CatBoostClassifier(verbose=0, class_weights=[2, 1, 2])
    catboost_params = {
        'iterations': [300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5]
    }
    catboost_search = GridSearchCV(catboost, catboost_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    catboost_search.fit(X_train[selected_features_rfe], y_train)
    print("📊 بهترین پارامترهای CatBoost:", catboost_search.best_params_)

    # بهینه‌سازی LightGBM
    lgbm = LGBMClassifier()
    lgbm_params = {
        'n_estimators': [300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'reg_lambda': [0, 1, 5]
    }
    lgbm_search = GridSearchCV(lgbm, lgbm_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    lgbm_search.fit(X_train[selected_features_rfe], y_train)
    print("📊 بهترین پارامترهای LightGBM:", lgbm_search.best_params_)

    # بهینه‌سازی XGBoost برای فاندامنتال
    xgboost = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgboost_params = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'reg_lambda': [1, 3, 5],
        'reg_alpha': [0, 1, 3]
    }
    xgboost_search = GridSearchCV(xgboost, xgboost_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    xgboost_search.fit(X_train_fund[selected_fund_features_rfe], y_train_fund)
    print("📊 بهترین پارامترهای XGBoost (فاندامنتال):", xgboost_search.best_params_)

    # تعریف مدل‌های بهینه‌شده
    catboost_model = CatBoostClassifier(**catboost_search.best_params_, verbose=100, class_weights=[2, 1, 2])
    lgbm_model = LGBMClassifier(**lgbm_search.best_params_)
    xgboost_model = XGBClassifier(**xgboost_search.best_params_, use_label_encoder=False, eval_metric='mlogloss')

    # ترکیب مدل‌ها (Ensemble)
    ensemble_model = VotingClassifier(estimators=[
        ('catboost', catboost_model),
        ('lightgbm', lgbm_model),
        ('xgboost_fundamental', xgboost_model)
    ], voting='soft', weights=[0.4, 0.4, 0.2])

    # آموزش مدل ترکیبی
    ensemble_model.fit(X_train[selected_features_rfe], y_train)
    y_pred_ensemble = ensemble_model.predict(X_test[selected_features_rfe])
    print("📊 گزارش عملکرد مدل ترکیبی:")
    print(classification_report(y_test, y_pred_ensemble))

    # ذخیره مدل
    joblib.dump(ensemble_model, 'models/ensemble_model_multi_rfe_with_fundamental.pkl')
    print("✅ مدل ذخیره شد: ensemble_model_multi_rfe_with_fundamental.pkl")

if __name__ == "__main__":
    main()