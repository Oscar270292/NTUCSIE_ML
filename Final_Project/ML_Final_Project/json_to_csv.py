'''
import pandas as pd
import json
import os

root_directory = r"C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\release"

df_dictionary = {}  # 用於存放結果的字典

for date_folder in os.listdir(root_directory):
    date_folder_path = os.path.join(root_directory, date_folder)

    if os.path.isdir(date_folder_path):
        for station_id in range(500101001, 500119092):  # 這裡使用 500119092 是因為 range() 不包含結束值
            station_filename = f"{station_id}.json"
            station_file_path = os.path.join(date_folder_path, station_filename)

            if os.path.exists(station_file_path):
                with open(station_file_path, 'r', encoding='utf-8') as file:
                    station_info = json.load(file)
                    df = pd.DataFrame(station_info)
                    df.index = [station_id] * len(df)
                    # 建立 DataFrame 的名稱，格式為 日期_編號_df
                    df_name = f"{date_folder}_{station_id}_df"
                    # 將 DataFrame 存入字典中，以日期_編號_df為鍵，對應到相應的 DataFrame
                    df_dictionary[df_name] = df
                print(df)
                '''