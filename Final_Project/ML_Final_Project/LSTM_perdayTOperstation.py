# 上禮拜對下禮拜一
import data_process
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense

station_id = "500101001"
week1_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231002.csv")
week2_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231009.csv")
week3_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231016.csv")
# week4_mon = data_process.read_data("20231023", station_id)
week5_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231030.csv")
week6_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231106.csv")
week7_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231113.csv")
week8_mon = pd.read_csv(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\20231120.csv")

week1_mon_nor = data_process.normalize(week1_mon)
week2_mon_nor = data_process.normalize(week2_mon)
week3_mon_nor = data_process.normalize(week3_mon)
# week4_mon_nor = data_process.normalize(week4_mon)
week5_mon_nor = data_process.normalize(week5_mon)
week6_mon_nor = data_process.normalize(week6_mon)
week7_mon_nor = data_process.normalize(week7_mon)
week8_mon_nor = data_process.normalize(week8_mon)
nor_list = [week1_mon_nor, week2_mon_nor, week3_mon_nor,  week5_mon_nor, week6_mon_nor, week7_mon_nor
    , week8_mon_nor]

x_train = []
y_train = []
x_train.append(data_process.build_trainx(week6_mon_nor))
y_train.append(data_process.build_trainy(week7_mon))
y_test = []
x_test = []
x_test.append(data_process.build_trainx(week7_mon_nor))
y_test.append(data_process.build_trainy(week8_mon))
# for i in range(6):
#     x_train.append(data_process.build_trainx(nor_list[i]))
#
# for j in range(1,7):
#     y_train.append(data_process.build_trainy(nor_list[j]))

x_train = np.array(x_train)
print(x_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_test = np.array(x_test)

scaler = MinMaxScaler(feature_range=(0, 1))
x_scale = scaler.fit_transform(np.array(x_train).reshape(-1, 1))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))
model = Sequential()
model.add(LSTM(4, input_shape=(x_train.shape[1], 5), return_sequences=True))
# model.add(Dropout(0.1))
model.add((Dense(1)))    # or use model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="adam")
model.summary()
history = model.fit(x_train, y_train, epochs=100, batch_size=12)

trainPredict = model.predict(x_test)
print(trainPredict)
# 回復預測資料值為原始數據的規模

trainPredict_2d = trainPredict.reshape(-1, trainPredict.shape[-1])
# trainPredict = scaler.inverse_transform(trainPredict_2d)
# trainY = scaler.inverse_transform(y_test)
# y_to_plot = y_train[0, :, 0]  # 选择第一个样本的所有时间步的数据


plt.plot(y_test.flatten(), color='blue', label='Predicted Google Stock Price')
plt.plot(trainPredict, color='red', label='Real Google Stock Price')  # 紅線表示真實股價# 藍線表示預測股價
plt.title('Ubike Prediction')
plt.xlabel('Time')
plt.ylabel('num of bike')
plt.legend()
plt.show()