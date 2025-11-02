import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

df = pd.read_csv("data/train.csv")
df = df.interpolate(method='linear')

target_cols = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
feature_cols = [col for col in df.columns if col not in target_cols]


X_all = df[feature_cols]
Y_all = df[target_cols]

X_all_train, X_all_test, Y_all_train, Y_all_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=4110)

models = {}
for target in target_cols:
    print(f"\nTraining model for {target}...")

    Y_train = Y_all_train[target]
    Y_test = Y_all_test[target]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=4110)
    model.fit(X_all_train, Y_train)

    models[target] = model
    
    Y_pred = model.predict(X_all_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"  MSE for {target}: {mse:.9e}")



index_to_drop = ['date_id']
cols_to_transform = [col for col in df.columns if col not in target_cols and col not in index_to_drop]

WINDOW_AVG = 21
WINDOW_SLOPE = 5

df_new_features = pd.DataFrame(index=df.index)

#needs to be improved to improve performance
for col in cols_to_transform:
    df_new_features[f'{col}_AVG_{WINDOW_AVG}'] = df[col].rolling(window=WINDOW_AVG, min_periods=1).mean()
    df_new_features[f'{col}_SLOPE_{WINDOW_SLOPE}'] = df[col] - df[col].shift(WINDOW_SLOPE)

df_rolling = pd.concat([df, df_new_features], axis=1)
df_rolling = df_rolling.drop('date_id', axis=1)
df_rolling = df_rolling.dropna()

X_all_train, X_all_test, Y_all_train, Y_all_test = train_test_split(X_all, Y_all, test_size=0.2, random_state=530)

models_trans = {}
for target in target_cols:
    print(f"\nTraining transformed model for {target}...")

    Y_train = Y_all_train[target]
    Y_test = Y_all_test[target]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=530)
    model.fit(X_all_train, Y_train)

    models_trans[target] = model
    
    Y_pred = model.predict(X_all_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"  MSE for {target}: {mse:.9e}")