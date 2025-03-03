import pandas as pd
import requests

# 下载数据文件
url = "https://www.csie.ntu.edu.tw/~htlin/course/ml23fall/hw1/hw1_train.dat"
response = requests.get(url)

# 将数据保存到临时文本文件
with open("hw1_train.txt", "wb") as file:
    file.write(response.content)

# 读取文本文件并分割数据
with open("hw1_train.txt", "r") as file:
    lines = file.readlines()

data = [line.split() for line in lines]

# 将数据保存为CSV文件
df = pd.DataFrame(data)
df.to_csv("hw1_train.csv", index=False, header=False)

print("成功hw1_train.csv")
