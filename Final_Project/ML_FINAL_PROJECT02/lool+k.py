# 上禮拜對下禮拜一
import os
import numpy as np
import pandas as pd
import data_process
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib as plt
import matplotlib.pyplot as plt
station = []

with open(r"C:/Users/GL66/PycharmProjects/ML_Final_Project/html.2023.final.data/sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
station = [item.strip() for sublist in station for item in sublist]
print(len(station))
pp = 0
for i in range(112):
    station_id = station[i]
    print(station_id)

    week1_mon = data_process.read_data("20231002", station_id)
    week2_mon = data_process.read_data("20231009", station_id)
    week3_mon = data_process.read_data("20231016", station_id)
    week4_mon = data_process.read_data("20231030", station_id)
    week5_mon = data_process.read_data("20231106", station_id)
    week6_mon = data_process.read_data("20231113", station_id)
    week7_mon = data_process.read_data("20231120", station_id)
    week8_mon = data_process.read_data("20231127", station_id)

    result = pd.concat([week1_mon, week2_mon, week3_mon,week4_mon,week5_mon,week6_mon,week7_mon])
    mean, big, small = result["sbi"].mean(), result["sbi"].max(), result["sbi"].min()
    week1_mon_nor = data_process.normalize(week1_mon.copy())
    week2_mon_nor = data_process.normalize(week2_mon.copy())
    week3_mon_nor = data_process.normalize(week3_mon.copy())
    week4_mon_nor = data_process.normalize(week4_mon.copy())
    week5_mon_nor = data_process.normalize(week5_mon.copy())
    week6_mon_nor = data_process.normalize(week6_mon.copy())
    week7_mon_nor = data_process.normalize(week7_mon.copy())
    week8_mon_nor = data_process.normalize(week8_mon.copy())

    nor_list = [week1_mon_nor, week2_mon_nor, week3_mon_nor, week4_mon_nor, week5_mon_nor, week6_mon_nor, week7_mon_nor,week8_mon_nor]
    x_train = []
    y_train = []
    x_test = [data_process.build_trainx(week7_mon_nor)]
    for i in range(len(nor_list) - 1):
        x_train.append(data_process.build_trainx(nor_list[i]))
    for i in range(1, len(nor_list)):
        y_train.append(data_process.build_trainy(nor_list[i]))
    print(x_train)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 8))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 8))

    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], 8), return_sequences=True))
    model.add(Dropout(0.01))
    model.add(LSTM(256, input_shape=(x_train.shape[1], 8), return_sequences=True))
    model.add(Dropout(0.01))
    model.add((Dense(1)))

    model.compile(loss='mean_squared_error', optimizer="adam")
    model.summary()
    model.fit(x_train, y_train, epochs=100, batch_size=5)
    lstm_pre = model.predict(x_test)
    lstm_pre = pd.DataFrame(lstm_pre.flatten(), columns=["sbi"])
    data_frame = data_process.denormalize(lstm_pre, big, small)

    if pp == 0:
        ddaa = data_frame
        pp = 20
    else:
        ddaa = pd.concat([ddaa, data_frame])

    print(i/112)

ddaa.to_csv('output2.csv', index=False)  # index=False 表示不写入索引

plt.plot(data_frame, color='blue', label='Predicted num')
plt.plot(week8_mon["sbi"], color='red', label='Real num')  # 紅線表示真實股價# 藍線表示預測股價
plt.title('Ubike Prediction')
plt.xlabel('Time')
plt.ylabel('num of bike')
plt.legend()
plt.show()

