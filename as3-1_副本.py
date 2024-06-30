import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fish_path = 'fish_data.npy'
data = np.load(fish_path, allow_pickle=True)

fish_data = data.item()
print(fish_data)

fish_lightness = fish_data['lightness']
fish_label = fish_data['label']

salmon_lightness = []
seabass_lightness = []

for i in range(len(fish_label)):
    if fish_label[i] == 'salmon':
        salmon_lightness.append(fish_lightness[i])
    elif fish_label[i] == 'sea_bass':
        seabass_lightness.append(fish_lightness[i])

salmon_lightness = np.array(salmon_lightness)
seabass_lightness = np.array(seabass_lightness)

print("Salmon lightness data:", salmon_lightness)
print("Seabass lightness data:", seabass_lightness)

# 找到数据的最大最小值
max_lightness = max(max(salmon_lightness), max(seabass_lightness))
min_lightness = min(min(salmon_lightness), min(seabass_lightness))

# 设置区间范围
bins = np.linspace(min_lightness, max_lightness, 21)  # 将数据分成20个区间
print("Bins:", bins)

# 计算每个区间的频数
salmon_hist = np.zeros(len(bins) - 1)
seabass_hist = np.zeros(len(bins) - 1)

for data in salmon_lightness:
    for i in range(len(bins) - 1):
        if bins[i] <= data < bins[i + 1]:
            salmon_hist[i] += 1
            break

for data in seabass_lightness:
    for i in range(len(bins) - 1):
        if bins[i] <= data < bins[i + 1]:
            seabass_hist[i] += 1
            break

# 打印频数分布以检查计算结果
print("Salmon histogram:", salmon_hist)
print("Seabass histogram:", seabass_hist)

# 绘制分布图
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4

ax.bar(bins[:-1] - width/2, salmon_hist, width=width, color='blue', label='Salmon')
ax.bar(bins[:-1] + width/2, seabass_hist, width=width, color='red', label='Seabass')

ax.set_xlabel('Lightness')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Lightness')
ax.legend()

plt.show()

# 定义阈值
T = 5.0
# 决策规则的公式：
# 如果 lightness < T, 分类为 'salmon'
# 否则, 分类为 'sea_bass'


def classify_fish(lightness, threshold):
    if lightness < threshold:
        return 'salmon'
    else:
        return 'sea_bass'


# 测试模型
test_lightness = [2.0, 6.0, 4.5, 5.5]
for lightness in test_lightness:
    result = classify_fish(lightness, T)
    print(f"Lightness: {lightness}, Classified as: {result}")


# 定义计算错误率的函数
def calculate_error_rate(threshold, salmon_lightness, seabass_lightness):
    errors = 0

    for lightness in salmon_lightness:
        if lightness >= threshold:
            errors += 1

    for lightness in seabass_lightness:
        if lightness < threshold:
            errors += 1

    total_samples = len(salmon_lightness) + len(seabass_lightness)
    error_rate = errors / total_samples
    return error_rate


# 定义阈值的范围
min_lightness = min(min(salmon_lightness), min(seabass_lightness))
max_lightness = max(max(salmon_lightness), max(seabass_lightness))
thresholds = np.linspace(min_lightness, max_lightness, 100)

# 计算每个阈值的错误率
error_rates = []
for threshold in thresholds:
    error_rate = calculate_error_rate(threshold, salmon_lightness, seabass_lightness)
    error_rates.append(error_rate)

# 找到最优阈值
optimal_threshold = thresholds[np.argmin(error_rates)]  # np.argmin, 找到数组中最小错误率
print(f"Optimal Threshold: {optimal_threshold}")

# 使用最优阈值进行分类
classified_results = [classify_fish(lightness, optimal_threshold) for lightness in fish_lightness]

# 显示分类结果
for i, lightness in enumerate(fish_lightness):
    print(f"Index: {i}, Lightness: {lightness}, Actual: {fish_label[i]}, Classified: {classified_results[i]}")

# 绘制错误率图
plt.figure(figsize=(10, 6))
plt.plot(thresholds, error_rates, marker='o')
plt.xlabel('Threshold')
plt.ylabel('Error Rate')
plt.title('Error Rate for Different Thresholds')
plt.show()

# 在表格中显示不同阈值的错误率
error_rate_table = pd.DataFrame({
    'Threshold': thresholds,
    'Error Rate': error_rates
})
print(error_rate_table)

