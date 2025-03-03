import matplotlib.pyplot as plt
import random

# 你的算法代码中给定的 C 值列表
C_values = [0.01, 0.1, 1, 10, 100]

# 通过重复的随机抽样验证过程，假设选择的 C 值
# 这里假设是随机选择的 C 值，实际情况根据验证结果选择最佳的 C 值
# 这里的代码只是示例，实际情况可能需要替换成你的算法逻辑
selections = []
for _ in range(1000):
    selected_C = random.choice(C_values)
    selections.append(selected_C)

# 计算每个 C 值的选择频率
selection_frequencies = [selections.count(c) for c in C_values]

# 设置离散的横轴刻度
index = list(range(len(C_values)))

plt.bar(index, selection_frequencies, align='center', alpha=0.7)
plt.xlabel('C values')
plt.ylabel('Selection Frequency')
plt.xticks(index, C_values)  # 设置横轴显示的特定刻度值
plt.title('C values vs Selection Frequency')
plt.show()