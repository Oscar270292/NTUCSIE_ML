from libsvm.svmutil import *
import random
import matplotlib.pyplot as plt
#用svm_read_problem 來讀取x_train和y_train
y, x = svm_read_problem('hw5_train.txt')
#用svm_read_problem 來讀取x_test和y_test
y_t, x_t = svm_read_problem('hw5_test.txt')
#當y=1時是1 其他都是-1
y_lab = [-1 if i != 1 else 1 for i in y]
y_lab_t = [-1 if j != 1 else 1 for j in y_t]
min_list = []
C_list = [0.01, 0.1, 1, 10, 100]
all_numbers = list(range(len(y)))
for h in range(1000):
    #隨機生成200個數來製作training set 其餘的做validating set
    random_numbers = random.sample(all_numbers, 200)
    remaining_numbers = [num for num in all_numbers if num not in random_numbers]
    #建立training set和validating set
    y_val = [y_lab[i] for i in random_numbers]
    x_val = [x[i] for i in random_numbers]
    y_train = [y_lab[j] for j in remaining_numbers]
    x_train = [x[j] for j in remaining_numbers]

    m1 = svm_train(y_train, x_train, '-s 0 -t 2 -g 1 -c 0.01 -q')
    m2 = svm_train(y_train, x_train, '-s 0 -t 2 -g 1 -c 0.1 -q')
    m3 = svm_train(y_train, x_train, '-s 0 -t 2 -g 1 -c 1 -q')
    m4 = svm_train(y_train, x_train, '-s 0 -t 2 -g 1 -c 10 -q')
    m5 = svm_train(y_train, x_train, '-s 0 -t 2 -g 1 -c 100 -q')

    p_labels, p_acc, p_vals = svm_predict(y_val, x_val, m1)
    p_labels2, p_acc2, p_vals2 = svm_predict(y_val, x_val, m2)
    p_labels3, p_acc3, p_vals3 = svm_predict(y_val, x_val, m3)
    p_labels4, p_acc4, p_vals4 = svm_predict(y_val, x_val, m4)
    p_labels5, p_acc5, p_vals5 = svm_predict(y_val, x_val, m5)
    #找best C
    acc_list = []
    acc_list.append(float(p_acc[0]))
    acc_list.append(float(p_acc2[0]))
    acc_list.append(float(p_acc3[0]))
    acc_list.append(float(p_acc4[0]))
    acc_list.append(float(p_acc5[0]))
    #找最小且表現最好的C
    max_value = max(acc_list)
    min_indices = [i for i, v in enumerate(acc_list) if v == max_value]
    i = min(min_indices)
    min_list.append(C_list[i])

selection_frequencies = [min_list.count(c) for c in C_list]
index = list(range(len(C_list)))

plt.bar(index, selection_frequencies, align='center', alpha=0.7)
plt.xlabel('C')
plt.ylabel('Frequency')
plt.xticks(index, C_list)
plt.title('C vs Frequency')
plt.show()