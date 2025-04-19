import os
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['OPENBLAS_NUM_THREADS'] = '12'

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize
from data.data_manager import get_candle_data
from feature_engineering_full_ultra_v2 import extract_features_full
from config import SYMBOLS

# تنظیم لاگ‌گیری برای ذخیره در فایل و نمایش در کنسول
os.makedirs('notebooks', exist_ok=True)

# تنظیم فرمت لاگ
log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

# تنظیم FileHandler برای ذخیره لاگ‌ها در فایل با کدگذاری UTF-8
file_handler = logging.FileHandler('notebooks/training_log.txt', encoding='utf-8')
file_handler.setFormatter(formatter)

# تنظیم StreamHandler برای نمایش لاگ‌ها در کنسول
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# تنظیم logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # حذف handlerهای پیش‌فرض
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 🎯 تنظیمات
TIMEFRAME = '4h'
WINDOW_SIZE = 100
LABEL_LOOKAHEAD = 12

def label_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("شروع لیبل‌گذاری داده‌ها...")
    df['future_return'] = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1
    df['label'] = 'Hold'
    df.loc[df['future_return'] > 0.005, 'label'] = 'Buy'
    df.loc[df['future_return'] < -0.005, 'label'] = 'Sell'
    logging.info("لیبل‌گذاری داده‌ها انجام شد.")
    return df.dropna()

def replace_outliers(df: pd.DataFrame, columns: list, threshold: float = 3) -> pd.DataFrame:
    logging.info("جایگزینی مقادیر پرت...")
    numeric_columns = [col for col in columns if df[col].dtype in [np.float64, np.int64]]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = winsorize(df[col], limits=[0.05, 0.05])
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers_count = (z_scores > threshold).sum()
            if outliers_count > 0:
                logging.info(f"تعداد مقادیر پرت در {col}: {outliers_count}")
    logging.info("مقادیر پرت مدیریت شدند.")
    return df

def ensure_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("اطمینان از نوع داده‌های مناسب...")
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(bool)
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                logging.warning(f"ستون {col} به عدد تبدیل نشد. به bool تبدیل می‌شود.")
                df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype(float)
    logging.info("نوع داده‌ها اصلاح شد.")
    return df

def plot_confusion_matrix(y_true, y_pred):
    logging.info("رسم ماتریس درهم‌ریختگی...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('notebooks/confusion_matrix.png')
    plt.close()
    logging.info("ماتریس درهم‌ریختگی ذخیره شد: notebooks/confusion_matrix.png")

def main():
    logging.info("شروع آموزش مدل...")
    all_data = []

    for symbol in SYMBOLS:
        logging.info(f"دریافت داده برای {symbol}")
        try:
            df = get_candle_data(symbol=symbol, limit=1000)
            if df.empty:
                logging.warning(f"داده‌ای برای {symbol} دریافت نشد.")
                continue
            logging.info(f"تعداد ردیف‌های داده برای {symbol}: {len(df)}")
            
            df['symbol'] = symbol
            logging.info(f"شروع استخراج فیچرها برای {symbol}...")
            df_feat = extract_features_full(df)
            logging.info(f"فیچرها برای {symbol} استخراج شدند: {df_feat.shape}")
            
            logging.info(f"شروع لیبل‌گذاری برای {symbol}...")
            df_labeled = label_data(df_feat)
            logging.info(f"لیبل‌گذاری برای {symbol} انجام شد: {df_labeled.shape}")
            
            df_labeled = df_labeled.copy()
            df_labeled['symbol'] = symbol
            all_data.append(df_labeled)
        except Exception as e:
            logging.error(f"خطا در پردازش {symbol}: {e}")
            continue

    if not all_data:
        logging.error("هیچ داده‌ای برای آموزش پیدا نشد.")
        return

    logging.info("ترکیب داده‌های تمام نمادها...")
    df_final = pd.concat(all_data)
    df_final.dropna(inplace=True)
    logging.info(f"داده‌های نهایی ساخته شدند: {df_final.shape}")
    
    if 'news_sentiment' not in df_final.columns:
        df_final['news_sentiment'] = 0.0

    total = len(df_final)
    non_null = df_final['news_sentiment'].notna().sum()
    percentage = (non_null / total) * 100
    logging.info(f"درصد داده‌هایی که news_sentiment دارند: {percentage:.2f}%  ({non_null}/{total})")

    fundamental_features = ['news_sentiment', 'fundamental_score', 'volume_score', 'news_score']
    technical_features = [col for col in df_final.columns if col not in fundamental_features + ['label', 'future_return', 'symbol']]

    df_final = replace_outliers(df_final, technical_features + fundamental_features)

    X = df_final.drop(columns=['label', 'future_return', 'symbol'])
    y = df_final['label']

    df_fundamental = df_final[df_final['news_sentiment'] != 0]
    X_fundamental = df_fundamental[fundamental_features]
    y_fundamental = df_fundamental['label']

    logging.info(f"ویژگی‌های تکنیکال: {technical_features}")
    logging.info(f"ویژگی‌های فاندامنتال: {fundamental_features}")
    logging.info(f"ابعاد داده‌های آموزشی (X): {X.shape}")
    logging.info(f"ابعاد داده‌های فاندامنتال (X_fundamental): {X_fundamental.shape}")

    if X.empty:
        logging.error("داده‌های آموزشی (X) خالی هستند. بررسی کنید که چرا داده‌ها بعد از پیش‌پردازش حذف شدند.")
        return

    label_mapping = {'Sell': 0, 'Hold': 1, 'Buy': 2}
    y = y.map(label_mapping)
    y_fundamental = y_fundamental.map(label_mapping)

    logging.info("تقسیم داده‌ها به آموزشی و آزمایشی...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"تقسیم داده‌ها انجام شد: X_train={X_train.shape}, X_test={X_test.shape}")

    if not X_fundamental.empty:
        X_train_fund, X_test_fund, y_train_fund, y_test_fund = train_test_split(X_fundamental, y_fundamental, test_size=0.2, random_state=42)
        logging.info(f"تقسیم داده‌های فاندامنتال انجام شد: X_train_fund={X_train_fund.shape}, X_test_fund={X_test_fund.shape}")
    else:
        logging.warning("داده کافی برای آموزش مدل فاندامنتال وجود ندارد.")

    if X_train.columns.duplicated().any():
        logging.warning(f"ستون‌های تکراری تو X_train: {X_train.columns[X_train.columns.duplicated()].tolist()}")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_train.columns.duplicated()]

    logging.info("چک کردن مقادیر نامناسب (NaN/Inf) در داده‌ها...")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)
    if not X_fundamental.empty:
        X_train_fund.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_train_fund.fillna(X_train_fund.mean(), inplace=True)
        X_test_fund.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_fund.fillna(X_test.mean(), inplace=True)
    logging.info("مقادیر نامناسب اصلاح شدند.")

    X_train = ensure_dtypes(X_train)
    X_test = ensure_dtypes(X_test)
    if not X_fundamental.empty:
        X_train_fund = ensure_dtypes(X_train_fund)
        X_test_fund = ensure_dtypes(X_test_fund)

    logging.info("اعمال SMOTE برای متعادل‌سازی کلاس‌ها...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"ابعاد داده‌های متعادل‌شده: X_train_resampled={X_train_resampled.shape}")

    logging.info("شروع بهینه‌سازی CatBoost...")
    catboost = CatBoostClassifier(verbose=0, thread_count=12)
    catboost_params = {
        'iterations': [300, 500],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [3, 5]
    }
    catboost_search = GridSearchCV(catboost, catboost_params, cv=3, scoring='f1_weighted', n_jobs=12)
    catboost_search.fit(X_train_resampled, y_train_resampled)
    logging.info("بهینه‌سازی CatBoost تموم شد.")
    logging.info(f"بهترین پارامترهای CatBoost: {catboost_search.best_params_}")

    logging.info("شروع بهینه‌سازی LightGBM...")
    lgbm = LGBMClassifier(min_child_samples=10, min_split_gain=0.01, class_weight='balanced')
    lgbm_params = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8],
        'reg_lambda': [0, 1],
        'feature_fraction': [0.8, 0.9]
    }
    lgbm_search = GridSearchCV(lgbm, lgbm_params, cv=3, scoring='f1_weighted', n_jobs=12)
    lgbm_search.fit(X_train_resampled, y_train_resampled)
    logging.info("بهینه‌سازی LightGBM تموم شد.")
    logging.info(f"بهترین پارامترهای LightGBM: {lgbm_search.best_params_}")

    if not X_fundamental.empty:
        logging.info("شروع بهینه‌سازی XGBoost (فاندامنتال)...")
        xgboost = XGBClassifier(eval_metric='mlogloss')
        xgboost_params = {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'reg_lambda': [1, 3],
            'reg_alpha': [0, 1]
        }
        xgboost_search = GridSearchCV(xgboost, xgboost_params, cv=3, scoring='f1_weighted', n_jobs=12)
        xgboost_search.fit(X_train_fund, y_train_fund)
        logging.info("بهینه‌سازی XGBoost تموم شد.")
        logging.info(f"بهترین پارامترهای XGBoost (فاندامنتال): {xgboost_search.best_params_}")
    else:
        logging.warning("XGBoost به دلیل نبود داده‌های فاندامنتال اجرا نمی‌شود.")

    logging.info("محاسبه اهمیت فیچرها...")
    best_catboost = catboost_search.best_estimator_
    importances = best_catboost.get_feature_importance()
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    logging.info("ویژگی‌های مهم (CatBoost):")
    logging.info(feature_importance_df.head(10))

    total_importance = feature_importance_df['importance'].sum()
    threshold = 0.01 * total_importance
    selected_features = feature_importance_df[feature_importance_df['importance'] > threshold]['feature']
    logging.info(f"تعداد فیچرهای منتخب: {len(selected_features)}")

    X_train_selected = X_train_resampled[selected_features]
    X_test_selected = X_test[selected_features]
    logging.info(f"ابعاد داده‌های منتخب: X_train_selected={X_train_selected.shape}, X_test_selected={X_test_selected.shape}")

    logging.info("تعریف مدل‌های بهینه‌شده...")
    catboost_model = CatBoostClassifier(**catboost_search.best_params_, verbose=100, class_weights=[2, 1, 2], thread_count=12)
    lgbm_model = LGBMClassifier(**lgbm_search.best_params_, min_child_samples=10, min_split_gain=0.01, class_weight='balanced')
    if not X_fundamental.empty:
        xgboost_model = XGBClassifier(**xgboost_search.best_params_, eval_metric='mlogloss')

    logging.info("آموزش مدل ترکیبی...")
    if not X_fundamental.empty:
        ensemble_model = VotingClassifier(estimators=[
            ('catboost', catboost_model),
            ('lightgbm', lgbm_model),
            ('xgboost_fundamental', xgboost_model)
        ], voting='soft', weights=[0.4, 0.4, 0.2])
    else:
        ensemble_model = VotingClassifier(estimators=[
            ('catboost', catboost_model),
            ('lightgbm', lgbm_model)
        ], voting='soft', weights=[0.5, 0.5])

    ensemble_model.fit(X_train_selected, y_train_resampled)
    logging.info("مدل ترکیبی آموزش دید.")
    
    y_pred_ensemble = ensemble_model.predict(X_test_selected)
    logging.info("گزارش عملکرد مدل ترکیبی:")
    logging.info(classification_report(y_test, y_pred_ensemble))

    plot_confusion_matrix(y_test, y_pred_ensemble)

    joblib.dump(ensemble_model, 'models/ensemble_model_multi_with_fundamental.pkl')
    logging.info("مدل ذخیره شد: ensemble_model_multi_with_fundamental.pkl")

if __name__ == "__main__":
    try:
        main()
        logging.info("آموزش مدل با موفقیت به پایان رسید.")
    except Exception as e:
        logging.error(f"خطای غیرمنتظره: {e}")
    finally:
        input("Press Enter to exit...")