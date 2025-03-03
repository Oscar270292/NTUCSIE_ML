from liblinear.liblinearutil import *
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

def e_val(x_val, y_val):
    x_train = []
    y_train = []
    if x_val == x_fold1:
        x_train = x_fold2 + x_fold3 + x_fold4 + x_fold5
        y_train = y_fold2 + y_fold3 + y_fold4 + y_fold5
    elif x_val == x_fold2:
        x_train = x_fold1 + x_fold3 + x_fold4 + x_fold5
        y_train = y_fold1 + y_fold3 + y_fold4 + y_fold5
    elif x_val == x_fold3:
        x_train = x_fold2 + x_fold1 + x_fold4 + x_fold5
        y_train = y_fold2 + y_fold1 + y_fold4 + y_fold5
    elif x_val == x_fold4:
        x_train = x_fold2 + x_fold3 + x_fold1 + x_fold5
        y_train = y_fold2 + y_fold3 + y_fold1 + y_fold5
    elif x_val == x_fold5:
        x_train = x_fold2 + x_fold3 + x_fold4 + x_fold1
        y_train = y_fold2 + y_fold3 + y_fold4 + y_fold1


    m1 = train(y_train, x_train, '-s 0 -c 500000 -e 0.000001 -q')
    m2 = train(y_train, x_train, '-s 0 -c 5000 -e 0.000001 -q')
    m3 = train(y_train, x_train, '-s 0 -c 50 -e 0.000001 -q')
    m4 = train(y_train, x_train, '-s 0 -c 0.5 -e 0.000001 -q')
    m5 = train(y_train, x_train, '-s 0 -c 0.005 -e 0.000001 -q')

    p1_labels, p1_acc, p1_vals = predict(y_val, x_val, m1)
    p2_labels, p2_acc, p2_vals = predict(y_val, x_val, m2)
    p3_labels, p3_acc, p3_vals = predict(y_val, x_val, m3)
    p4_labels, p4_acc, p4_vals = predict(y_val, x_val, m4)
    p5_labels, p5_acc, p5_vals = predict(y_val, x_val, m5)

    acc = np.array([])
    acc = np.append(acc, float(p1_acc[0]))
    acc = np.append(acc, float(p2_acc[0]))
    acc = np.append(acc, float(p3_acc[0]))
    acc = np.append(acc, float(p4_acc[0]))
    acc = np.append(acc, float(p5_acc[0]))
    return acc


max_list = []
for u in range(128):
    np.random.seed(u)
    #將0-199隨機分成5組數字
    data = np.arange(0, 200)
    np.random.shuffle(data)
    split_data = np.array_split(data, 5)
    subset_1, subset_2, subset_3, subset_4, subset_5 = split_data

    x_fold1 = []
    y_fold1 = []
    for i in subset_1:
        x_fold1.append(phi_3_x(x_train_matrix[i, :]))
        y_fold1.append(y_train_all[i])

    x_fold2 = []
    y_fold2 = []
    for i in subset_2:
        x_fold2.append(phi_3_x(x_train_matrix[i, :]))
        y_fold2.append(y_train_all[i])

    x_fold3 = []
    y_fold3 = []
    for i in subset_3:
        x_fold3.append(phi_3_x(x_train_matrix[i, :]))
        y_fold3.append(y_train_all[i])

    x_fold4 = []
    y_fold4 = []
    for i in subset_4:
        x_fold4.append(phi_3_x(x_train_matrix[i, :]))
        y_fold4.append(y_train_all[i])

    x_fold5 = []
    y_fold5 = []
    for i in subset_5:
        x_fold5.append(phi_3_x(x_train_matrix[i, :]))
        y_fold5.append(y_train_all[i])

    acc_tot = e_val(x_fold1, y_fold1)
    acc_tot +=e_val(x_fold2, y_fold2)
    acc_tot +=e_val(x_fold3, y_fold3)
    acc_tot +=e_val(x_fold4, y_fold4)
    acc_tot +=e_val(x_fold5, y_fold5)

    max_value = max(acc_tot)
    max_indices = [i for i, v in enumerate(acc_tot) if v == max_value]
    i = max(max_indices)

    lambda_list = [-6, -4, -2, 0, 2]
    max_list.append(lambda_list[i])

plt.hist(max_list, bins=20)
plt.xlabel('log10(λ*)')
plt.ylabel('Times')
plt.title('Distribution of log10(λ*)')
plt.show()