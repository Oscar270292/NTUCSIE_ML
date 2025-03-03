import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def read_data(date_folder, station_id):
    # 讀取資料
    root_directory = r"C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\release"
    file_path = os.path.join(root_directory, date_folder, f"{station_id}.json")
    data = pd.read_json(file_path)
    csv_file_path = station_id + '.csv'
    data.to_csv(csv_file_path, index=False)
    train = pd.read_csv(csv_file_path)
    df = train.T
    df = df.fillna(method='backfill', inplace=False)
    df = df.fillna(method='ffill', inplace=False)
    # df = nan(df)
    # 每20分鐘取一個點
    df_filtered = df.iloc[::20]
    df_filtered = df_filtered.reset_index(names="datetime")
    df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])
    df_filtered['minute_of_day'] = df_filtered['datetime'].dt.hour * 60 + df_filtered['datetime'].dt.minute
    date = datetime.strptime(date_folder, '%Y%m%d')
    df_filtered["dayofweek"] = date.weekday()
    df_filtered.drop('datetime', axis=1, inplace=True)
    df_filtered = df_filtered.set_axis(['tot', 'sbi', 'bemp', "act", "min", "dayofweek"], axis=1)
    df_filtered.drop('bemp', axis=1, inplace=True)

    if date_folder[3:6]== '10':
        temp = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202310-Temperature.csv")
        rain = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202310-rain.csv")

    elif (date_folder[3:6] == '11'):
        temp = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202311-Temperature.csv")
        rain = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202311-rain.csv")

    else:
        temp = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202312-Temperature.csv")
        rain = pd.read_csv("C:/Users/GL66/PycharmProjects/ML_FINAL_PROJECT02/202312-rain.csv")

    date = date_folder[-2:]
    # 提取名为 column_name 的列
    temp_data = list(temp.loc[int(date)-1])  # 放入日期-1
    rain_data = list(rain.loc[int(date) - 1])  # 放入日期-1

    repeat = []
    repeat_rain = []
    for i in range(1, 25):
        if(temp_data[i]== '--'):
            temp_data[i] = 0
        if(rain_data[i] == '--'):
            rain_data[i] = 0
        repeat.extend(np.tile(float(temp_data[i]), 3))
        repeat_rain.extend(np.tile(float(rain_data[i]), 3))
    expanded_data = pd.DataFrame(repeat, columns=['temp'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    expanded_data = pd.DataFrame(repeat_rain, columns=['rain'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    expanded_data = pd.DataFrame(repeat, columns=['holiday'])
    df_filtered = pd.concat([df_filtered, expanded_data], axis=1)
    df_filtered['holiday'] = 0
    if date_folder == '20231009':
        df_filtered['holiday'] = 1
    return df_filtered


def nan(train):
    train = train.T
    for index, row in train.iterrows():
        for col in range(len(train.columns)):
            # 檢查是否為NaN
            if pd.isna(row[col]):
                # 獲取左邊和右邊的值
                left = row[col - 1] if col > 0 else np.nan
                right = row[col + 1] if col < len(train.columns) - 1 else np.nan
                # 計算平均值，忽略NaN
                avg = np.nanmean([left, right])
                # 填充NaN值
                train.iat[index, col] = avg
    return train.T


def normalize(df):
    columns_to_normalize = ['sbi', 'bemp', "min"]
    for column in columns_to_normalize:
        min_val = df[column].min()
        max_val = df[column].max()
        if min_val != max_val:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    return df


def build_trainx(df):
    return np.array(df)




def build_trainy(df):
    return np.array(df["sbi"])


def denormalize(df, max, min):
    if max != min:
        df["sbi"] = (df["sbi"] * (max - min) + min)
    return df


def id_list(day, mc):
    start_date = datetime(2023, 10, day)
    end_date = datetime(2023, 10, day, 23, 40)
    interval = timedelta(minutes=20)
    middle_code = mc
    current_date = start_date
    date_list = []
    while current_date <= end_date:
        formatted_date = current_date.strftime("%Y%m%d_%H:%M")
        formatted_datetime = f"202310{day}_{middle_code}_{formatted_date.split('_')[-1]}"
        date_list.append(formatted_datetime)
        current_date += interval
    return date_list