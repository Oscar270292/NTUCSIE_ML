import pandas as pd
import data_process
import numpy as np
from keras.models import Sequential
from keras.src.layers import Dropout
from keras.layers import LSTM
from keras.layers import Dense
from datetime import datetime, timedelta

station = []
with open(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
flat_list = [item.strip() for sublist in station for item in sublist]
final_df =  pd.DataFrame(columns=["id", "sbi"])
for jj in flat_list:
    def id_list(day, mc):


        # 定义开始日期和结束日期
        start_date = datetime(2023, 12, day)
        end_date = datetime(2023, 12, day, 23, 40)

        # 定义时间间隔
        interval = timedelta(minutes=20)

        # 固定中间编号
        middle_code = mc

        # 生成时间列表
        current_date = start_date
        date_list = []

        while current_date <= end_date:
            formatted_date = current_date.strftime("%Y%m%d_%H:%M")
            formatted_datetime = f"202312{day}_{middle_code}_{formatted_date.split('_')[-1]}"
            date_list.append(formatted_datetime)
            current_date += interval
        return date_list




    station_id = jj

    week1_mon = data_process.read_data("20231008", station_id)
    week2_mon = data_process.read_data("20231015", station_id)
    week3_mon = data_process.read_data("20231029", station_id)
    week5_mon = data_process.read_data("20231105", station_id)
    week6_mon = data_process.read_data("20231112", station_id)
    week7_mon = data_process.read_data("20231119", station_id)
    week8_mon = data_process.read_data("20231126", station_id)
    result = pd.concat([week1_mon, week2_mon, week3_mon, week5_mon, week6_mon, week7_mon, week8_mon])
    mean, big, small = result["sbi"].mean(), result["sbi"].max(), result["sbi"].min()
    week1_mon_nor = data_process.normalize(week1_mon.copy())
    week2_mon_nor = data_process.normalize(week2_mon.copy())
    week3_mon_nor = data_process.normalize(week3_mon.copy())
    week5_mon_nor = data_process.normalize(week5_mon.copy())
    week6_mon_nor = data_process.normalize(week6_mon.copy())
    week7_mon_nor = data_process.normalize(week7_mon.copy())
    week8_mon_nor = data_process.normalize(week8_mon.copy())
    nor = data_process.normalize(result)
    nor_list = [week1_mon_nor, week2_mon_nor, week3_mon_nor, week5_mon_nor, week6_mon_nor, week7_mon_nor
        , week8_mon_nor]
    x_train = []
    y_train = []
    x_test = [data_process.build_trainx(week8_mon_nor)]
    for i in range(len(nor_list) - 1):
        x_train.append(data_process.build_trainx(nor_list[i]))
    for i in range(1, len(nor_list)):
        y_train.append(data_process.build_trainy(nor_list[i]))
    print(len(x_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))

    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], 5), return_sequences=True))
    model.add(Dropout(0.01))
    model.add(LSTM(256, input_shape=(x_train.shape[1], 5), return_sequences=True))
    model.add(Dropout(0.01))
    model.add((Dense(1)))

    model.compile(loss='mean_squared_error', optimizer="adam")
    model.summary()
    model.fit(x_train, y_train, epochs=300, batch_size=5)
    lstm_pre = model.predict(x_test)
    lstm_pre = pd.DataFrame(lstm_pre.flatten(), columns=["sbi"])
    ll = data_process.denormalize(lstm_pre, big, small)

    data = {'id': id_list(10, station_id), 'sbi': ll['sbi']}
    df = pd.DataFrame(data)
    final_df= pd.concat([final_df, df])
print(final_df)

final_df.to_csv('1210.csv', index=False)