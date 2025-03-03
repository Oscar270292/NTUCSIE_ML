import numpy as np
import matplotlib.pyplot as plt
#建立一個sigmoid計算方法的函式
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
Eout_A_ZO = np.array([])
Eout_B_ZO = np.array([])
for t in range(128):
    #設置random seed
    np.random.seed(t)
    N_train = 256
    N_test = 4096

    #隨機生成y train和y test的值
    y_train = np.random.choice([-1, 1], N_train)
    y_test = np.random.choice([-1, 1], N_test)

    #y = +1 時x值的分布
    mean_plus1 = [3, 2]
    cov_plus1 = [[0.4, 0], [0, 0.4]]

    #y = -1 時x值的分布
    mean_minus1 = [5, 0]
    cov_minus1 = [[0.6, 0], [0, 0.6]]

    #生成x train值
    X_train = np.array([])
    for i in y_train:
        if i == 1:
            x_plus1 = np.random.multivariate_normal(mean_plus1, cov_plus1)
            x_plus1 = np.insert(x_plus1, 0, 1)
            X_train = np.append(X_train, x_plus1)
        else:
            x_minus1 = np.random.multivariate_normal(mean_minus1, cov_minus1)
            x_minus1 = np.insert(x_minus1, 0, 1)
            X_train = np.append(X_train, x_minus1)
    #生成x test值
    X_test = np.array([])
    for i in y_test:
        if i == 1:
            x_plus1 = np.random.multivariate_normal(mean_plus1, cov_plus1)
            x_plus1 = np.insert(x_plus1, 0, 1)
            X_test = np.append(X_test, x_plus1)
        else:
            x_minus1 = np.random.multivariate_normal(mean_minus1, cov_minus1)
            x_minus1 = np.insert(x_minus1, 0, 1)
            X_test = np.append(X_test, x_minus1)
    #將x值變成matrix
    X_train_matrix = X_train.reshape(-1, 3)
    X_test_matrix = X_test.reshape(-1, 3)
    #產生x的pseudo inverse
    x_pseudo_inv = np.linalg.pinv(X_train_matrix)

    #Eout_A的計算方法
    W_A = x_pseudo_inv @ y_train
    # mis中的值會小於0如果ys小於0
    mis = W_A @ X_test_matrix.T * y_test
    negative_val = np.sum(mis < 0)
    Eout_A = negative_val / N_test

    # Eout_B的計算方法
    W_B = np.zeros(3)
    for k in range(500):
        delta_Ein = 0
        for j in range(256):
            one_delta_Ein = sigmoid(-1 * y_train[j] * ( X_train_matrix[j] @ W_B.T)) * (-1 * y_train[j] * X_train_matrix[j])
            delta_Ein += one_delta_Ein
        delta_Ein = delta_Ein / N_train
        W_B = W_B - 0.1 * delta_Ein
    # mis_B中的值會小於0如果ys小於0
    mis_B = W_B @ X_test_matrix.T * y_test
    negative_val_B = np.sum(mis_B < 0)
    Eout_B = negative_val_B / N_test

    Eout_A_ZO = np.append(Eout_A_ZO, Eout_A)
    Eout_B_ZO = np.append(Eout_B_ZO, Eout_B)

#印出 Eout median
medianEout_A = np.median(Eout_A_ZO)
print("Median for Eout_A:", medianEout_A)
medianEout_B = np.median(Eout_B_ZO)
print("Median for Eout_B:", medianEout_B)
#畫圖
x = Eout_A_ZO
y = Eout_B_ZO
plt.scatter(x, y, label='Data Points', color='b', marker='o')
plt.xlabel('Eout_A_ZO')
plt.ylabel('Eout_B_ZO')
plt.title('Scatter Plot for (Eout_A, Eout_B)')
plt.legend()
plt.show()