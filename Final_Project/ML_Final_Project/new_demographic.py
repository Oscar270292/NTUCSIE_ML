import json
import os
from datetime import datetime

# 定义 sarea 到数字的映射
sarea_mapping = {
    "大安區": 1,
    "大同區": 2,
    "士林區": 3,
    "文山區": 4,
    "中正區": 5,
    "中山區": 6,
    "內湖區": 7,
    "北投區": 8,
    "松山區": 9,
    "南港區": 10,
    "信義區": 11,
    "萬華區": 12,
    "臺大公館校區": 13
}

# 读取站点基本信息
with open('html.2023.final.data/demographic.json', 'r', encoding='utf-8') as file:
    station_info = json.load(file)

# 获取最新日期的文件夹名称
release_folder = 'html.2023.final.data/release'
latest_date = max(os.listdir(release_folder))
latest_folder = os.path.join(release_folder, latest_date)

# 更新站点信息
new_station_info = {}
for station_id, info in station_info.items():
    file_path = os.path.join(latest_folder, f'{station_id}.json')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            latest_data = json.load(file)
            last_time = max(latest_data.keys(),
                            key=lambda x: datetime.strptime(x, "%H:%M")
                            if latest_data[x] else datetime.min)
            last_data = latest_data[last_time]

            new_station_info[station_id] = {
                "sarea": sarea_mapping[info["sarea"]],
                "lat": info["lat"],
                "lng": info["lng"],
                "tot": last_data.get("tot", 0),
                "act": int(last_data.get("act", 0))
            }

# 创建 data_transform 文件夹（如果不存在）
os.makedirs('data_transform', exist_ok=True)

# 保存新的站点信息
with open('data_transform/new_demographic.json', 'w', encoding='utf-8') as file:
    json.dump(new_station_info, file, indent=4)

print("New station information updated and saved.")
