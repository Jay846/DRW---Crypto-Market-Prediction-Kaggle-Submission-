import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
import xgboost as xgb
import lightgbm as lgb
import gc

SEED = 2024
np.random.seed(SEED)

print("Loading Data...")
train = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/drw-crypto-market-prediction/sample_submission.csv')

def engineer_features(df):
    df = df.copy()
    df['spread'] = np.abs(df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-6)
    df['ofi'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-6)
    df['buy_sell_ratio'] = (df['buy_qty'] + 1e-6) / (df['sell_qty'] + 1e-6)
    df['volume_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-6)
    df['log_volume'] = np.log1p(df['volume'])
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def simple_rolling(df, features, windows=[3, 5], lags=[1]):
    for col in features:
        for w in windows:
            df[f'{col}_rollmean{w}'] = df[col].rolling(w).mean()
            df[f'{col}_ema{w}'] = df[col].ewm(span=w, adjust=False).mean()
        for l in lags:
            df[f'{col}_lag{l}'] = df[col].shift(l)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

print("Feature Engineering...")
train = engineer_features(train)
test = engineer_features(test)

core_feats = ['spread', 'ofi', 'buy_sell_ratio', 'volume_imbalance']
print(f"Using core features: {core_feats}")

train = add_simple_rolling(train, core_feats)
test = add_simple_rolling(test, core_feats)

ignore = set(['timestamp', 'asset', 'label', 'log_return_forward_1s'])
feature_cols = [col for col in train.columns if col not in ignore and not train[col].isnull().all()]

min_lag = 5
train = train.iloc[min_lag:].reset_index(drop=True)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train[feature_cols])
X_test_scaled = scaler.transform(test[feature_cols])

y_raw = train['label']
y = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-6)

# Time-based split
N = len(X_train_scaled)
split = int(0.85 * N)
gap = int(0.05 * N)
X_tr = X_train_scaled[:split]
y_tr = y.iloc[:split]
X_val = X_train_scaled[split+gap:]
y_val = y.iloc[split+gap:]

del train
gc.collect()

print("Training Models...")

# Model
model_xgb = xgb.XGBRegressor(
    objective='reg:squarederror',
    learning_rate=0.012,
    max_depth=5,
    n_estimators=600,
    subsample=0.85,
    colsample_bytree=0.75,
    tree_method='hist',
    reg_alpha=0.4,
    reg_lambda=1.5,
    early_stopping_rounds=30,
    random_state=SEED,
    verbosity=0
)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
xgb_val = model_xgb.predict(X_val)

# LightGBM
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    learning_rate=0.01,
    max_depth=6,
    n_estimators=600,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.3,
    reg_lambda=1.2,
    max_bin=128,
    force_col_wise=True,
    random_state=SEED
)
model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse')
lgb_val = model_lgb.predict(X_val)

# Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
ridge_val = ridge.predict(X_val)

# Bayesian Ridge
bayes = BayesianRidge()
bayes.fit(X_tr, y_tr)
bayes_val = bayes.predict(X_val)


w_xgb = 0.4
w_lgb = 0.3
w_ridge = 0.2
w_bayes = 0.1

val_blend = w_xgb*xgb_val + w_lgb*lgb_val + w_ridge*ridge_val + w_bayes*bayes_val
val_corr = np.corrcoef(y_val, val_blend)[0, 1]
print("Validation Pearson Correlation (blended):", round(val_corr, 6))

del X_tr, X_val, y_tr, y_val
gc.collect()

print("Predicting on Test Set...")
xgb_pred = model_xgb.predict(X_test_scaled)
lgb_pred = model_lgb.predict(X_test_scaled)
ridge_pred = ridge.predict(X_test_scaled)
bayes_pred = bayes.predict(X_test_scaled)

test_preds = w_xgb*xgb_pred + w_lgb*lgb_pred + w_ridge*ridge_pred + w_bayes*bayes_pred
test_preds_final = test_preds * y_raw.std() + y_raw.mean()

print("Saving Submission...")
submission = sample_submission.copy()
submission['prediction'] = test_preds_final
submission.to_csv('submission.csv', index=False)
print("submission.csv saved with", len(submission), "rows.")
