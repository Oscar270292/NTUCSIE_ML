from liblinear.liblinearutil import *
import pandas as pd

train_data = pd.read_csv("hw4_train.csv")
train_data = pd.DataFrame(train_data)

x_train_matrix = train_data.iloc[:, :6].to_numpy()
y_train = train_data.iloc[:, -1].tolist()

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

x_train = []
for i in range(200):
    x_train.append(phi_3_x(x_train_matrix[i, :]))

#C = 1 / 2λ
m1 = train(y_train, x_train, '-s 0 -c 500000 -e 0.000001')
m2 = train(y_train, x_train, '-s 0 -c 5000 -e 0.000001')
m3 = train(y_train, x_train, '-s 0 -c 50 -e 0.000001')
m4 = train(y_train, x_train, '-s 0 -c 0.5 -e 0.000001')
m5 = train(y_train, x_train, '-s 0 -c 0.005 -e 0.000001')

p1_labels, p1_acc, p1_vals = predict(y_train, x_train, m1)
p2_labels, p2_acc, p2_vals = predict(y_train, x_train, m2)
p3_labels, p3_acc, p3_vals = predict(y_train, x_train, m3)
p4_labels, p4_acc, p4_vals = predict(y_train, x_train, m4)
p5_labels, p5_acc, p5_vals = predict(y_train, x_train, m5)

print(p1_acc)
print(p2_acc)
print(p3_acc)
print(p4_acc)
print(p5_acc)

