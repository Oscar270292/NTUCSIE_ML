from libsvm.svmutil import *
#用svm_read_problem 來讀取x_train和y_train
y, x = svm_read_problem('hw5_train.txt')
#用svm_read_problem 來讀取x_test和y_test
y_t, x_t = svm_read_problem('hw5_test.txt')
#當y=1時是1 其他都是-1
y_lab = [-1 if i != 1 else 1 for i in y]
y_lab_t = [-1 if j != 1 else 1 for j in y_t]
m1 = svm_train(y_lab, x, '-s 0 -t 2 -g 1 -d 2 -c 0.01')
p_labels, p_acc, p_vals = svm_predict(y_lab_t, x_t, m1)
m2 = svm_train(y_lab, x, '-s 0 -t 2 -g 1 -d 2 -c 0.1')
p_labels2, p_acc2, p_vals2 = svm_predict(y_lab_t, x_t, m2)
m3 = svm_train(y_lab, x, '-s 0 -t 2 -g 1 -d 2 -c 1')
p_labels3, p_acc3, p_vals3 = svm_predict(y_lab_t, x_t, m3)
m4 = svm_train(y_lab, x, '-s 0 -t 2 -g 1 -d 2 -c 10')
p_labels4, p_acc4, p_vals4 = svm_predict(y_lab_t, x_t, m4)
m5 = svm_train(y_lab, x, '-s 0 -t 2 -g 1 -d 2 -c 100')
p_labels5, p_acc5, p_vals5 = svm_predict(y_lab_t, x_t, m5)