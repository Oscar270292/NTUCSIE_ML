import random
import numpy as np
import matplotlib.pyplot as plt

eout_minus_ein = np.array([])
ein_fin = np.array([])
eout_fin = np.array([])
#To run 2000 times
for i in range(2000):
    # To generate a data including x and y
    data = []
    x_r = np.random.uniform(-1, 1, 8)
    x = np.sort(x_r)
    noise = np.random.choice([1, -1], 8, p=[0.9, 0.1])
    y = np.sign(x) * noise
    # Generate theta
    theta = np.array([])
    for i in range(len(x)-1):
        theta_val = (x[i] + x[i+1])/2
        theta = np.append(theta, theta_val)
    theta = np.insert(theta, 0, -1)
    s = np.array([-1, 1])
    #random pick s and theta
    random_s = np.random.choice(s)
    random_theta = np.random.choice(theta)
    i = random_theta
    j = random_s
    y_h = np.sign(x - i) * j
    # mis is an array that become -1 when there's an error
    mis = y * y_h
    ein = np.count_nonzero(mis == -1) / 8

    eout = 0.5 - 0.4 * j + 0.4 * j * abs(i)
    eout_minus_ein = np.append(eout_minus_ein, eout - ein)
    ein_fin = np.append(ein_fin, ein)
    eout_fin = np.append(eout_fin, eout)

print(eout_minus_ein)
print(ein_fin)
print(eout_fin)

med = np.median(eout_minus_ein)
print(f"Median: {med}")
#draw a scatter plot
plt.scatter(ein_fin, eout_fin, c='b', marker='o', label='Data Points')
plt.xlabel('Ein(g)')
plt.ylabel('Eout(g)')
plt.title('Scatter Plot of (Ein(g), Eout(g))')
plt.legend()
plt.show()