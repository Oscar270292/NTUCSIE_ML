from datetime import datetime, timedelta

# 定义开始日期和结束日期
start_date = datetime(2023, 12, 6)
end_date = datetime(2023, 12, 6, 23, 40)

# 定义时间间隔
interval = timedelta(minutes=20)

# 固定中间编号
middle_code = "500101001"

# 生成时间列表
current_date = start_date
date_list = []

while current_date <= end_date:
    formatted_date = current_date.strftime("%Y%m%d_%H:%M")
    formatted_datetime = f"20231206_{middle_code}_{formatted_date.split('_')[-1]}"
    date_list.append(formatted_datetime)
    current_date += interval


station = []
with open(r"C:\Users\GL66\PycharmProjects\ML_Final_Project\html.2023.final.data\sno_test_set.txt", 'r', encoding='utf-8') as file:
    lines = file.readlines()
    station.append(lines)
flat_list = [item.strip() for sublist in station for item in sublist]

print(len(flat_list))
