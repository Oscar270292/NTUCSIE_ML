import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
#from keras.layers.normalization import BatchNormalization
#import matplotlib.pyplot as plt
import json
import os
from keras.layers.normalization import BatchNormalization
# import matplotlib.pyplot as plt
import json
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, TimeDistributed
from keras.models import Sequential


def normalize(train):
  train = train.drop(["Date"], axis=1)
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

date_folder = '20231002'
date1_folder = '20231009'
station_id = '500101001'
root_directory = r"C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\release"

file_path = os.path.join(root_directory, date_folder, f"{station_id}.json")
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        station_info = json.load(file)
keys_to_delete = [key for key, value in station_info.items() if not value]

for key in keys_to_delete:
    del station_info[key]

x = []
for time_key in station_info.keys():
    hour, minute = map(int, time_key.split(':'))  # 將時間字串分割成小時和分鐘
    total_minutes = 60 * hour + minute  # 轉換成總共的分鐘數
    x.append(total_minutes)

y = []
for key, value in station_info.items():
    y.append(value['sbi'])

data = pd.DataFrame({'time': x, 'amount': y})
print(data)
def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

def buildOneToOneModel(shape):
  model = Sequential()
  model.add(LSTM(10, input_length=shape[1], input_dim=shape[1],return_sequences=True))
  # output shape: (1, 1)
  model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

X_train = data['time'].values
y_train = data['amount'].values

X_train, Y_train = shuffle(X_train, y_train)
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

model = buildOneToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
