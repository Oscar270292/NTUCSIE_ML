import json
import os
import pandas as pd
import numpy as np

data_folder_path = r'C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\release'  # 將路徑指定為存放資料的資料夾
data = []


date_folder = '20231127'
station_id = '500101001'

file_path = os.path.join(data_folder_path, date_folder, f"{station_id}.json")
if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        station_info = json.load(file)
        for time, info in station_info.items():
            hour, minute = map(int, time.split(':'))  # 將時間字串分割成小時和分鐘
            total_minutes = 60 * hour + minute  # 轉換成總共的分鐘數
            data.append({
                '時間': total_minutes,
                'tot': info.get('tot', None),
                'sbi': info.get('sbi', None),
                'bemp': info.get('bemp', None),
                'act': info.get('act', None)
            })

df = pd.DataFrame(data)
print(df)
df['act'] = df['act'].astype(float)
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
fill_missing_values(df['act'])
#df['act'].fillna(1, inplace=True)
print(df)
df.to_csv('20231127.csv', index=False)