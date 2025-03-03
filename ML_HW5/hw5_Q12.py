from libsvm.svmutil import *
import numpy as np
import matplotlib.pyplot as plt

#用svm_read_problem 來讀取x_train和y_train
y, x = svm_read_problem('hw5_train.txt')
y_lab = [-1 if i != 3 else 1 for i in y]
C_list = [0.01, 0.1, 1, 10, 100]
C_loglist = [-2, -1, 0, 1, 2]
w_list = []

for i in range(5):
    c = C_list[i]
    m = svm_train(y_lab, x, f'-s 0 -t 2 -g 1 -d 2 -c {c}')
    #用get_sv_indices和get_sv_coef來找alpha
    s = m.get_sv_indices()
    k = m.get_sv_coef()
    n = len(x[0])
    w = [0] * n
    for i in range(len(s)):
        alpha = k[i][0]
        sv = x[s[i] - 1]
        for j in range(1, n + 1):
            if j in sv:
                w[j - 1] += alpha * sv[j]
    #計算w的長度
    w_vector= np.array(w)
    w_length = np.linalg.norm(w_vector)
    w_list.append(w_length)

plt.plot(C_loglist, w_list, marker='o')
plt.xlabel('logC')
plt.ylabel('||w||')
plt.title('logC vs ||w||')
plt.grid(True)
plt.show()