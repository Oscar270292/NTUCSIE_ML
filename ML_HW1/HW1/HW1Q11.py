#此份程式碼大約需跑30分鐘
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# train_data 是我們的df
train_data = pd.read_csv("hw1_train.csv")
train_data = pd.DataFrame(train_data)

# 整理欄名並且把會不見的第一列重新裝回去
original_columns = train_data.columns.tolist()
num_columns = len(train_data.columns)

new_columns = list(range(num_columns))
train_data.columns = new_columns

train_data.loc[-1] = original_columns
train_data.index = train_data.index + 1
train_data = train_data.sort_index()
train_data.iloc[0, :] = train_data.iloc[0, :].astype(float)

#一個可以求出 UPdate Time的函式
def GetUpdateTime():
    w0 = np.zeros((13, 1))
    row = random.randint(0, 255)
    x_list = train_data.iloc[row, :-1].tolist()
    x_list.insert(0, 11.26)
    x = np.array([x_list])
    x = x.T
    y = train_data.iloc[row, -1]
    w1 = w0 + y * x
    correct_tm = 0
    ans = 0

    while correct_tm < 5*256:
        row = random.randint(0, 255)
        x_list = train_data.iloc[row, :-1].tolist()
        x_list.insert(0, 11.26)
        x = np.array([x_list])
        x = x.T
        y = int(train_data.iloc[row, -1])
        check_value = np.dot(w1.T, x)
        if check_value > 0 and y == -1:
            ans += 1
            w1 = w1 + y * x
            correct_tm = 0
        elif check_value <= 0 and y == 1:
            ans += 1
            w1 = w1 + y * x
            correct_tm = 0
        else:
            correct_tm += 1
    return ans

his_init = {}
his_fin = {}
#把所有答案裝進his_init字典裡 key是update time, value是這個update time的次數
for i in range(1000):
    answer = GetUpdateTime()
    if answer in his_init:
        his_init[answer] += 1
    else:
        his_init[answer] = 1
    print(i)#print出可以知道執行到第幾次
#按照key的大小依序放入his_fin字典裡
sorted_keys = sorted(his_init.keys())
for key in sorted_keys:
    his_fin[key] = his_init[key]
his_fin = {str(k): v for k, v in his_fin.items()}
#整理一個his_draw 將百位數相同的his_fin的key都視為同個key
his_draw = {}
xpnt = 0
for key in sorted_keys:
    xpnt = (key // 100) * 100
    if xpnt in his_draw:
        his_draw[xpnt] += 1
    else:
        his_draw[xpnt] = 1
#計算中位數
n = 0
mid1 = 0
mid2 = 0
for key in his_fin:
    n += his_fin[key]
    if n >= 500:
        mid1 = int(key)
        break
n = 0
for key in his_fin:
    n += his_fin[key]
    if n >= 501:
        mid2 = int(key)
        break
print((mid1 + mid2)/2)
#畫圖
x_positions = list(his_draw.keys())
data_points = list(his_draw.values())
plt.bar(x_positions, data_points, width=100, align='edge')
plt.xlabel('Update Time')
plt.ylabel('Number')
plt.title('The Distribution of the Number of Updates')
plt.show()