import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFE
import joblib
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.stats.mstats import winsorize
from data.data_manager import get_candle_data
from feature_engineering_full_ultra_v2 import extract_features_full
from config import SYMBOLS

os.makedirs('notebooks', exist_ok=True)

log_format = '%(asctime)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)

file_handler = logging.FileHandler('notebooks/training_log.txt', encoding='utf-8')
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)

TIMEFRAME = '4h'
WINDOW_SIZE = 100
LABEL_LOOKAHEAD = 12

def label_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("شروع لیبل‌گذاری داده‌ها...")
    df['future_return'] = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1
    df['label'] = 'Hold'
    df.loc[df['future_return'] > 0.002, 'label'] = 'Buy'  # کاهش آستانه به 0.002
    df.loc[df['future_return'] < -0.002, 'label'] = 'Sell'
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

def plot_confusion_matrix(y_true, y_pred, model_name):
    logging.info(f"رسم ماتریس درهم‌ریختگی برای {model_name}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'notebooks/confusion_matrix_{model_name}.png')
    plt.close()
    logging.info(f"ماتریس درهم‌ریختگی ذخیره شد: notebooks/confusion_matrix_{model_name}.png")

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

    fundamental_features = [f"news_emb_{i}" for i in range(768)] + ['volume_score']
    technical_features = [col for col in df_final.columns if col not in fundamental_features + ['label', 'future_return', 'symbol']]

    df_final = replace_outliers(df_final, technical_features + fundamental_features)

    X = df_final.drop(columns=['label', 'future_return', 'symbol'])
    y = df_final['label']

    logging.info(f"ویژگی‌های تکنیکال: {technical_features}")
    logging.info(f"ویژگی‌های فاندامنتال: {fundamental_features}")
    logging.info(f"ابعاد داده‌های آموزشی (X): {X.shape}")

    if X.empty:
        logging.error("داده‌های آموزشی (X) خالی هستند. بررسی کنید که چرا داده‌ها بعد از پیش‌پردازش حذف شدند.")
        return

    label_mapping = {'Sell': 0, 'Hold': 1, 'Buy': 2}
    y = y.map(label_mapping)

    logging.info("تقسیم داده‌ها به آموزشی و آزمایشی...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"تقسیم داده‌ها انجام شد: X_train={X_train.shape}, X_test={X_test.shape}")

    if X_train.columns.duplicated().any():
        logging.warning(f"ستون‌های تکراری تو X_train: {X_train.columns[X_train.columns.duplicated()].tolist()}")
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_train.columns.duplicated()]

    logging.info("چک کردن مقادیر نامناسب (NaN/Inf) در داده‌ها...")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)
    logging.info("مقادیر نامناسب اصلاح شدند.")

    X_train = ensure_dtypes(X_train)
    X_test = ensure_dtypes(X_test)

    logging.info("اعمال SMOTE برای متعادل‌سازی کلاس‌ها...")
    logging.info(f"توزیع کلاس‌ها قبل از SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"توزیع کلاس‌ها بعد از SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    logging.info(f"ابعاد داده‌های متعادل‌شده: X_train_resampled={X_train_resampled.shape}")

    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in class_counts.items()}
    logging.info(f"وزن‌های محاسبه‌شده برای کلاس‌ها: {class_weights}")

    logging.info("شروع بهینه‌سازی CatBoost...")
    catboost = CatBoostClassifier(verbose=0, thread_count=-1)
    catboost_params = {
        'iterations': [300, 500],
        'learning_rate': [0.05, 0.1],
        'depth': [4, 6],
        'l2_leaf_reg': [3, 5]
    }
    catboost_search = GridSearchCV(catboost, catboost_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    catboost_search.fit(X_train_resampled, y_train_resampled)
    logging.info("بهینه‌سازی CatBoost تموم شد.")
    logging.info(f"بهترین پارامترهای CatBoost: {catboost_search.best_params_}")
    joblib.dump(catboost_search.best_estimator_, 'models/catboost_model.pkl')
    logging.info("مدل CatBoost ذخیره شد: models/catboost_model.pkl")

    y_pred_catboost = catboost_search.predict(X_test)
    logging.info("گزارش عملکرد CatBoost:")
    logging.info(classification_report(y_test, y_pred_catboost))
    plot_confusion_matrix(y_test, y_pred_catboost, "CatBoost")

    logging.info("شروع بهینه‌سازی LightGBM...")
    lgbm = LGBMClassifier(min_child_samples=10, min_split_gain=0.01)
    lgbm_params = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [6, 8],
        'reg_lambda': [0, 1],
        'feature_fraction': [0.8, 0.9]
    }
    lgbm_search = GridSearchCV(lgbm, lgbm_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    lgbm_search.fit(X_train_resampled, y_train_resampled)
    logging.info("بهینه‌سازی LightGBM تموم شد.")
    logging.info(f"بهترین پارامترهای LightGBM: {lgbm_search.best_params_}")
    joblib.dump(lgbm_search.best_estimator_, 'models/lightgbm_model.pkl')
    logging.info("مدل LightGBM ذخیره شد: models/lightgbm_model.pkl")

    y_pred_lgbm = lgbm_search.predict(X_test)
    logging.info("گزارش عملکرد LightGBM:")
    logging.info(classification_report(y_test, y_pred_lgbm))
    plot_confusion_matrix(y_test, y_pred_lgbm, "LightGBM")

    logging.info("شروع بهینه‌سازی XGBoost...")
    xgboost = XGBClassifier(eval_metric='mlogloss')
    xgboost_params = {
        'n_estimators': [200, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'reg_lambda': [1, 3],
        'reg_alpha': [0, 1]
    }
    xgboost_search = GridSearchCV(xgboost, xgboost_params, cv=3, scoring='f1_weighted', n_jobs=-1)
    xgboost_search.fit(X_train_resampled, y_train_resampled)
    logging.info("بهینه‌سازی XGBoost تموم شد.")
    logging.info(f"بهترین پارامترهای XGBoost: {xgboost_search.best_params_}")
    joblib.dump(xgboost_search.best_estimator_, 'models/xgboost_model.pkl')
    logging.info("مدل XGBoost ذخیره شد: models/xgboost_model.pkl")

    y_pred_xgboost = xgboost_search.predict(X_test)
    logging.info("گزارش عملکرد XGBoost:")
    logging.info(classification_report(y_test, y_pred_xgboost))
    plot_confusion_matrix(y_test, y_pred_xgboost, "XGBoost")

    logging.info("اجرای RFE برای انتخاب ویژگی‌ها...")
    estimator = CatBoostClassifier(**catboost_search.best_params_, verbose=0, thread_count=-1)
    rfe = RFE(estimator, n_features_to_select=50)  # انتخاب 50 ویژگی
    rfe.fit(X_train_resampled, y_train_resampled)
    rfe_selected_features = X_train_resampled.columns[rfe.support_].tolist()
    logging.info(f"ویژگی‌های انتخاب‌شده توسط RFE: {rfe_selected_features}")

    X_train_selected = X_train_resampled[rfe_selected_features]
    X_test_selected = X_test[rfe_selected_features]
    logging.info(f"ابعاد داده‌های منتخب: X_train_selected={X_train_selected.shape}, X_test_selected={X_test_selected.shape}")

    logging.info("تعریف مدل‌های بهینه‌شده...")
    catboost_model = CatBoostClassifier(**catboost_search.best_params_, verbose=100, class_weights=class_weights, thread_count=-1)
    lgbm_model = LGBMClassifier(**lgbm_search.best_params_, min_child_samples=10, min_split_gain=0.01, class_weight=class_weights)
    xgboost_model = XGBClassifier(**xgboost_search.best_params_, eval_metric='mlogloss', scale_pos_weight=class_weights)

    logging.info("آموزش مدل ترکیبی...")
    ensemble_model = VotingClassifier(estimators=[
        ('catboost', catboost_model),
        ('lightgbm', lgbm_model),
        ('xgboost', xgboost_model)
    ], voting='soft', weights=[0.4, 0.4, 0.2])

    ensemble_model.fit(X_train_selected, y_train_resampled)
    logging.info("مدل ترکیبی آموزش دید.")
    
    y_pred_ensemble = ensemble_model.predict(X_test_selected)
    logging.info("گزارش عملکرد مدل ترکیبی:")
    logging.info(classification_report(y_test, y_pred_ensemble))
    plot_confusion_matrix(y_test, y_pred_ensemble, "Ensemble")

    joblib.dump(ensemble_model, 'models/ensemble_model_multi_with_fundamental_Pro.pkl')
    logging.info("مدل ذخیره شد: ensemble_model_multi_with_fundamental.pkl")

if __name__ == "__main__":
    try:
        main()
        logging.info("آموزش مدل با موفقیت به پایان رسید.")
    except Exception as e:
        logging.error(f"خطای غیرمنتظره: {e}")
    finally:
        input("Press Enter to exit...")