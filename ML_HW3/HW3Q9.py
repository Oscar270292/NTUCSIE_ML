import numpy as np
import matplotlib.pyplot as plt

Ein_sqr = np.array([])
for t in range(128):
    #設置random seed
    np.random.seed(t)

    N = 256
    #隨機生成y值
    y_train = np.random.choice([-1, 1], N)

    #y = +1 時x值的分布
    mean_plus1 = [3, 2]
    cov_plus1 = [[0.4, 0], [0, 0.4]]

    #y = -1 時x值的分布
    mean_minus1 = [5, 0]
    cov_minus1 = [[0.6, 0], [0, 0.6]]

    #生成x值
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
    #將x值變成matrix
    X_train_matrix = X_train.reshape(-1, 3)
    #產生x的pseudo inverse
    x_pseudo_inv = np.linalg.pinv(X_train_matrix)
    W = x_pseudo_inv @ y_train
    #先計算X * W - Y的矩陣在平方相加取平均
    Ein_mat = X_train_matrix @ W - y_train
    Ein = np.sum(Ein_mat**2) / N
    Ein_sqr = np.append(Ein_sqr, Ein)
#印出 Ein median
medianEin = np.median(Ein_sqr)
print("Median:", medianEin)

#把histogram畫出來
plt.hist(Ein_sqr, bins=20)
plt.xlabel('Ein_sqr')
plt.ylabel('Time')
plt.title('Distribution of Ein_sqr')
plt.show()