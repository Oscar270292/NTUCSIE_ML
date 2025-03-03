import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
#from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
# 資料夾路徑
data_folder_path = r'C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\release'  # 將路徑指定為存放資料的資料夾

# 日期範圍
start_date = '20231025'
end_date = '20231126'

# 站點編號
station_id = '500101001'

data = []

# 遍歷日期範圍內的資料夾
for folder_name in os.listdir(data_folder_path):
    if start_date <= folder_name <= end_date:
        folder_path = os.path.join(data_folder_path, folder_name)

        # 檢查資料夾是否存在以站點編號命名的 JSON 檔案
        station_file_path = os.path.join(folder_path, f'{station_id}.json')
        if os.path.exists(station_file_path):
            with open(station_file_path, 'r') as file:
                station_data = json.load(file)


                for time, info in station_data.items():
                    hour, minute = map(int, time.split(':'))  # 將時間字串分割成小時和分鐘
                    total_minutes = 60 * hour + minute  # 轉換成總共的分鐘數
                    data.append({
                        '日期': folder_name,
                        '時間': total_minutes,
                        'tot': info.get('tot', None),
                        'sbi': info.get('sbi', None),
                        'bemp': info.get('bemp', None),
                        'act': info.get('act', None)
                    })

df = pd.DataFrame(data)

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

# 填充空值的处理函数
def fill_missing_values(column):
    # 处理首尾空值
    if pd.notnull(column.iloc[0]):
        column.iloc[0] = column.iloc[0]
    else:
        column.iloc[0] = column.iloc[1]

    if pd.notnull(column.iloc[-1]):
        column.iloc[-1] = column.iloc[-1]
    else:
        column.iloc[-1] = column.iloc[-2]

    # 处理中间空值
    column.fillna((column.ffill() + column.bfill()) / 2, inplace=True)

# 对特定列应用处理函数
fill_missing_values(df['tot'])
fill_missing_values(df['sbi'])
fill_missing_values(df['bemp'])
#fill_missing_values(df['act'])
df['act'].fillna(1, inplace=True)
print(df)

# 將DataFrame中每20行的第一行留下，其餘刪除
df_filtered = df.iloc[::20]

# 重新設置索引
df_filtered.reset_index(drop=True, inplace=True)

print(df_filtered)

df_filtered['日期'] = df_filtered['日期'].astype(int)
df_filtered['act'] = df_filtered['act'].astype(int)
df_filtered.info()

columns_to_normalize = df_filtered.drop(['日期', 'tot', 'act'], axis=1).copy()
# 初始化正規化器
scaler = MinMaxScaler()

# 對除日期、tot、act 之外的所有列進行正規化
#normalized_data = scaler.fit_transform(columns_to_normalize)

# 將正規化後的數據轉換為 DataFrame
#columns_to_normalize_normalized = pd.DataFrame(normalized_data, columns=columns_to_normalize.columns)
columns_to_normalize_normalized = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns)
# 將 '日期'、'tot'、'act' 列添加回新的 DataFrame
columns_to_normalize_normalized['日期'] = df_filtered['日期']
columns_to_normalize_normalized['tot'] = df_filtered['tot']
columns_to_normalize_normalized['act'] = df_filtered['act']
# 輸出正規化後的 DataFrame
print(columns_to_normalize_normalized)

X_train = []
y_train = []

# 按照规定生成 X_train 和 y_train
for i in range(25):
    start_X = i * 72
    end_X = (i + 3) * 72
    X_train.append(columns_to_normalize_normalized.iloc[start_X:end_X].values.flatten())

    start_y = (7 + i) * 72
    end_y = (10 + i) * 72
    y_train.append(columns_to_normalize_normalized.iloc[start_y:end_y]['sbi'].values.tolist())

X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train:")
print(len(X_train))
print("y_train:")
print(y_train)
print(len(y_train[24]))
print(y_train[24])
print(len(y_train[23]))
print(y_train[0])
print(type(y_train))
X_train, Y_train, X_val, Y_val = splitData(X_train, y_train, 0.2)


def build_model(input_length, input_dim):
    d = 0.3
    model = Sequential()
    model.add(LSTM(128, input_shape=(input_length, input_dim), return_sequences=True))

    model.add(Dropout(d))

    model.add(LSTM(64, input_shape=(input_length, input_dim), return_sequences=False))

    model.add(Dropout(d))

    model.add(Dense(1, activation='softmax'))
    # linear / softmax(多分類) / sigmoid(二分法)

    model.compile(loss='mse', optimizer='adam')
    return model

Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]
X_train = X_train.reshape(-1, 1296, X_train.shape[1])
my_callbacks = [
tf.keras.callbacks.EarlyStopping(patience=300, monitor = 'val_loss')] ######## 在訓練組訓練，使用驗證組選取
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min',save_best_only=True)
call_backlist = [my_callbacks,checkpoint]
lstm00 = build_model(1296,20)
historylstm0 = lstm00.fit( X_train, Y_train, batch_size=40,shuffle=False , epochs=1000,validation_data=(X_val,Y_val),callbacks=call_backlist)
lstm00.summary()
"""
# 选择满足条件的日期范围
dates_to_include = [20031002, 20031008, 20031009, 20031018]

# 过滤符合条件的数据
filtered_df = columns_to_normalize_normalized[columns_to_normalize_normalized['日期'].isin(dates_to_include) | ((columns_to_normalize_normalized['日期'] >= 20231025))]

# 根据日期每三天一个单位生成 X_train
X_train = []
y_train = []

for i in range(0, len(filtered_df) - 2, 3):
    X_train.append(filtered_df.iloc[i:i + 3][['日期', '時間', 'tot', 'sbi', 'bemp', 'act']].values.flatten())
    y_train.append(filtered_df.iloc[i + 2]['sbi'])

X_train = np.array(X_train)
y_train = np.array(y_train)

print("X_train:")
print(X_train)
print("y_train:")
print(y_train)
print(len(X_train))
print(X_train[3])
print(X_train[2])
print(X_train[1])
print(len(y_train))
"""

"""
columns_to_normalize = df_filtered.drop(['日期', 'tot', 'act'], axis=1).copy()

# 初始化正規化器
scaler = MinMaxScaler()

# 對除日期、tot、act 之外的所有列進行正規化
normalized_data = scaler.fit_transform(columns_to_normalize)

# 將正規化後的數據轉換為 DataFrame
columns_to_normalize_normalized = pd.DataFrame(normalized_data, columns=columns_to_normalize.columns)

# 將 '日期'、'tot'、'act' 列添加回新的 DataFrame
columns_to_normalize_normalized['日期'] = df_filtered['日期']
columns_to_normalize_normalized['tot'] = df_filtered['tot']
columns_to_normalize_normalized['act'] = df_filtered['act']

# 輸出正規化後的 DataFrame
print(columns_to_normalize_normalized)
def buildManyToOneModel(shape):
  model = Sequential()
  model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

X_train = columns_to_normalize_normalized['sbi'].to_numpy()
Y_train = columns_to_normalize_normalized.drop('sbi', axis=1).values



scaler_1d = MinMaxScaler()
X_train = scaler_1d.fit_transform(X_train.reshape(-1, 1))
scaler_2d = MinMaxScaler()
for i in range(1, Y_train.shape[1]):
    Y_train[:, i] = scaler_2d.fit_transform(Y_train[:, i].reshape(-1, 1)).flatten()
"""

""""X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.125)
print(X_train)
print(Y_train)
print(X_val)
print(Y_val)

Y_train = Y_train[:,:,np.newaxis]
Y_val = Y_val[:,:,np.newaxis]

model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

"""