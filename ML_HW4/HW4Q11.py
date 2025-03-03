from liblinear.liblinearutil import *
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train_data = pd.read_csv("hw4_train.csv")
train_data = pd.DataFrame(train_data)

x_train_matrix = train_data.iloc[:, :6].to_numpy()
y_train_all = train_data.iloc[:, -1].tolist()

def phi_3_x(x):
    d = len(x)
    features = []

    features.append(1)
    features.extend(x)
    for i in range(d):
        for j in range(i, d):
            features.append(x[i] * x[j])
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                features.append(x[i] * x[j] * x[k])
    return features

max_list = []
for u in range(128):
    np.random.seed(u)
    x_train = []
    y_train = []
    random_numbers = random.sample(range(0, 200), 120)
    for i in random_numbers:
        x_train.append(phi_3_x(x_train_matrix[i, :]))
        y_train.append(y_train_all[i])

    x_val = []
    y_val = []
    all_numbers = list(range(0, 200))
    remaining_numbers = [num for num in all_numbers if num not in random_numbers]
    for k in remaining_numbers:
        x_val.append(phi_3_x(x_train_matrix[k, :]))
        y_val.append(y_train_all[k])
    # C = 1 / 2λ
    m1 = train(y_train, x_train, '-s 0 -c 500000 -e 0.000001')
    m2 = train(y_train, x_train, '-s 0 -c 5000 -e 0.000001')
    m3 = train(y_train, x_train, '-s 0 -c 50 -e 0.000001')
    m4 = train(y_train, x_train, '-s 0 -c 0.5 -e 0.000001')
    m5 = train(y_train, x_train, '-s 0 -c 0.005 -e 0.000001')

    p1_labels, p1_acc, p1_vals = predict(y_val, x_val, m1)
    p2_labels, p2_acc, p2_vals = predict(y_val, x_val, m2)
    p3_labels, p3_acc, p3_vals = predict(y_val, x_val, m3)
    p4_labels, p4_acc, p4_vals = predict(y_val, x_val, m4)
    p5_labels, p5_acc, p5_vals = predict(y_val, x_val, m5)

    acc_list = []
    acc_list.append(float(p1_acc[0]))
    acc_list.append(float(p2_acc[0]))
    acc_list.append(float(p3_acc[0]))
    acc_list.append(float(p4_acc[0]))
    acc_list.append(float(p5_acc[0]))

    max_value = max(acc_list)
    max_indices = [i for i, v in enumerate(acc_list) if v == max_value]
    i = max(max_indices)

    lambda_list = [-6, -4, -2, 0, 2]
    max_list.append(lambda_list[i])

plt.hist(max_list, bins=20)
plt.xlabel('log10(λ*)')
plt.ylabel('Times')
plt.title('Distribution of log10(λ*)')
plt.show()