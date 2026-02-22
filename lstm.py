import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- 1. データの読み込みと前処理 (Daily) ---
file_path_daily = "./drive/MyDrive/Colab Notebooks/追試/Hyytiala_COSfluxes_meteo_2013-2017_daily.csv"

try:
    data_daily = pd.read_csv(
        file_path_daily,
        parse_dates=['datetime'],
        index_col='datetime'
    )
    print(f"日次ファイルの読み込み成功。元の形状: {data_daily.shape}")
except FileNotFoundError:
    print(f"エラー: ファイル '{file_path_daily}' が見つかりません。")
    exit()

# 論文（4.4節）に基づき、2014年のデータを除外
data_daily = data_daily[data_daily['Year'] != 2014]

# 【重要】論文で使用された入力変数
# 論文(表1)では 'Ta', 'Ts'(SoilTA), 'SWC'(SoilWA), 'VPD', 'PAR', 'LAI' に加え、
# 'LST', 'FAPAR', 'NDVI', 'EVI' といった衛星データも使用されています。
#
# 今回のCSVには衛星データ（LAIを除く）が含まれていません。
# これが、論文のR²(0.86)と結果が異なる(0.59)最も可能性の高い原因です。
#
# ここでは、CSVに含まれる変数のみを使用します。
FEATURE_COLUMNS_DAILY = ['Ta', 'SoilTA', 'SoilWA', 'VPD', 'PAR', 'LAI']
TARGET_COLUMN = 'FCOS_gapfilled'

data_subset_daily = data_daily[FEATURE_COLUMNS_DAILY + [TARGET_COLUMN]].copy()

# 欠損値の処理
data_subset_daily.fillna(method='ffill', inplace=True)
data_subset_daily.fillna(method='bfill', inplace=True)

if data_subset_daily.empty:
    print("エラー: 日次データの処理後にデータが0件になりました。")
    exit()

print(f"日次データ（2014年除外・欠損処理後）の形状: {data_subset_daily.shape}")

# --- 2. スケーリング (Daily) ---
scaler_features_daily = MinMaxScaler(feature_range=(0, 1))
scaler_target_daily = MinMaxScaler(feature_range=(0, 1))

scaled_features_daily = scaler_features_daily.fit_transform(data_subset_daily[FEATURE_COLUMNS_DAILY])
scaled_target_daily = scaler_target_daily.fit_transform(data_subset_daily[[TARGET_COLUMN]])

# --- 3. 時系列データセットの作成 (Daily) ---
# 論文 表2 (Daily, COS): Memory Length = 20
LOOK_BACK_DAILY = 20

def create_dataset(features, target, look_back=1):
    X, Y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:(i + look_back), :])
        Y.append(target[i + look_back, 0])
    return np.array(X), np.array(Y)

X_daily, Y_daily = create_dataset(scaled_features_daily, scaled_target_daily, LOOK_BACK_DAILY)

print(f"日次シーケンスデータ - X: {X_daily.shape}, Y: {Y_daily.shape}")

# --- 4. データの分割 (Daily) ---
# 論文 (2.3.2節) に基づき、時系列で 70% (学習) と 30% (テスト) に分割
X_train_daily, X_test_daily, Y_train_daily, Y_test_daily = train_test_split(
    X_daily, Y_daily, test_size=0.3, shuffle=False
)

print(f"日次 学習データ: {X_train_daily.shape}")
print(f"日次 テストデータ: {X_test_daily.shape}")

# --- 5. LSTMモデルの構築 (Daily, COS) ---
# 論文 表2 (Daily, COS):
# Batch Size: 64, Learning Rate: 0.01, Weight Decay: 0.0006
model_daily = Sequential()
model_daily.add(LSTM(
    units=50,
    activation='relu',
    input_shape=(X_train_daily.shape[1], X_train_daily.shape[2]),
    kernel_regularizer=l2(0.0006) # Weight Decay
))
model_daily.add(Dropout(0.2))
model_daily.add(Dense(units=1))

optimizer_daily = Adam(learning_rate=0.01)
model_daily.compile(optimizer=optimizer_daily, loss='mean_squared_error')

model_daily.summary()

# --- 6. モデルの学習 (Daily) ---
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history_daily = model_daily.fit(
    X_train_daily, Y_train_daily,
    epochs=100,
    batch_size=64, # 論文 表2
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# --- 7. モデルの評価 (Daily) ---
Y_pred_scaled_daily = model_daily.predict(X_test_daily)

Y_pred_daily = scaler_target_daily.inverse_transform(Y_pred_scaled_daily)
Y_test_orig_daily = scaler_target_daily.inverse_transform(Y_test_daily.reshape(-1, 1))

rmse_daily = np.sqrt(mean_squared_error(Y_test_orig_daily, Y_pred_daily))
r2_daily = r2_score(Y_test_orig_daily, Y_pred_daily)

print("\n--- 日次モデル評価結果 (FCOS) ---")
print(f"RMSE: {rmse_daily:.4f}")
print(f"R² (決定係数): {r2_daily:.4f}")
print(f"（論文のR² (参考): 0.86）")
print("R²が論文と異なる主な理由は、入力変数（衛星データ）の不足と考えられます。")

# --- 8. 可視化 (Daily) ---
plt.figure(figsize=(15, 6))
plt.plot(Y_test_orig_daily, label='Actual FCOS (観測値)', color='blue')
plt.plot(Y_pred_daily, label='Predicted FCOS (予測値)', color='red', alpha=0.7)
plt.title('日次 FCOSフラックス予測 (Daily Model)')
plt.ylabel('FCOS_gapfilled (pmol m⁻² s⁻¹) ')
plt.legend()
plt.show()