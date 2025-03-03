import pandas as pd
from libsvm.svmutil import *
#用svm_read_problem 來讀取x和y
y, x = svm_read_problem('hw5_train.txt')
# 當y=4時是1 其他都是-1
y_lab = [-1 if i != 4 else 1 for i in y]
m1 = svm_train(y_lab, x, '-s 0 -t 1 -d 2 -g 1 -r 1 -c 0.1')
print('=====================m1==========================')
m2 = svm_train(y_lab, x, '-s 0 -t 1 -d 3 -g 1 -r 1 -c 0.1')
print('=====================m2==========================')
m3 = svm_train(y_lab, x, '-s 0 -t 1 -d 4 -g 1 -r 1 -c 0.1')
print('=====================m3==========================')
m4 = svm_train(y_lab, x, '-s 0 -t 1 -d 2 -g 1 -r 1 -c 1')
print('=====================m4==========================')
m5 = svm_train(y_lab, x, '-s 0 -t 1 -d 3 -g 1 -r 1 -c 1')
print('=====================m5==========================')
m6 = svm_train(y_lab, x, '-s 0 -t 1 -d 4 -g 1 -r 1 -c 1')
print('=====================m6==========================')
m7 = svm_train(y_lab, x, '-s 0 -t 1 -d 2 -g 1 -r 1 -c 10')
print('=====================m7==========================')
m8 = svm_train(y_lab, x, '-s 0 -t 1 -d 3 -g 1 -r 1 -c 10')
print('=====================m8==========================')
m9 = svm_train(y_lab, x, '-s 0 -t 1 -d 4 -g 1 -r 1 -c 10')
print('=====================m9==========================')
