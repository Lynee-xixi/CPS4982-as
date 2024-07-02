import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
iris_path = 'IrisData.csv'
iris_data = pd.read_csv(iris_path)

# 确认读取的数据，前5行
print(iris_data.head())

# 提取Versicolor和Virginica类别的数据
versicolor_virginica = iris_data[(iris_data['Species'] == 'Iris-versicolor') | (iris_data['Species'] == 'Iris-virginica')]

# 确认提取的数据
print(versicolor_virginica.head())  # 显示Iris-versicolor或Iris-virginica样本的前5行
print(versicolor_virginica.shape)

# 提取花瓣长度和花萼宽度特征
features = versicolor_virginica[['PetalLengthCm', 'SepalWidthCm']]
labels = versicolor_virginica['Species']

print(features.head())
print(labels.head())

# 分离训练集和测试集
train_data = []
test_data = []

for label in ['Iris-versicolor', 'Iris-virginica']:
    category_data = versicolor_virginica[versicolor_virginica['Species'] == label]
    train_data.append(category_data.iloc[:40])
    test_data.append(category_data.iloc[40:50])

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# 确认训练集和测试集数据
print("Training set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

# 确认训练集和测试集的特征和标签
train_features = train_data[['PetalLengthCm', 'SepalWidthCm']]
train_labels = train_data['Species']
test_features = test_data[['PetalLengthCm', 'SepalWidthCm']]
test_labels = test_data['Species']

print(train_features.head())
print(train_labels.head())
print(test_features.head())
print(test_labels.head())

# 绘制训练集样本分布
plt.figure(figsize=(10, 6))

for label, color, marker in zip(['Iris-versicolor', 'Iris-virginica'], ['blue', 'red'], ['o', 'x']):
    subset = train_data[train_data['Species'] == label]
    plt.scatter(subset['PetalLengthCm'], subset['SepalWidthCm'], c=color, label=label, marker=marker)

plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.title('Training Set Distribution')
plt.legend()
plt.show()


# 定义线性模型
def linear_model(x, k, b):
    return k * x + b


# 分类决策规则
def classify(features, k, b):
    return np.where(features['PetalLengthCm'] * k + b - features['SepalWidthCm'] >= 0, 'Iris-virginica', 'Iris-versicolor')


# 计算分类错误数量
def calculate_errors(features, labels, k, b):
    predictions = classify(features, k, b)
    errors = np.sum(predictions != labels)
    return errors


# 给定k和b的组合，计算训练集和测试集的分类错误数量
k, b = 1, 0  # 初始值，可以根据需要调整
train_errors = calculate_errors(train_features, train_labels, k, b)
test_errors = calculate_errors(test_features, test_labels, k, b)

print(f"Training Errors: {train_errors}")
print(f"Testing Errors: {test_errors}")

# 画出分类直线
plt.figure(figsize=(10, 6))

for label, color, marker in zip(['Iris-versicolor', 'Iris-virginica'], ['blue', 'red'], ['o', 'x']):
    subset = train_data[train_data['Species'] == label]
    plt.scatter(subset['PetalLengthCm'], subset['SepalWidthCm'], c=color, label=label, marker=marker)

x_vals = np.linspace(train_features['PetalLengthCm'].min(), train_features['PetalLengthCm'].max(), 100)
y_vals = linear_model(x_vals, k, b)
plt.plot(x_vals, y_vals, color='green', linestyle='--', label=f'Linear Classifier: y = {k}x + {b}')

plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.title('Training Set with Linear Classifier')
plt.legend()
plt.show()


# 搜索最佳 k 和 b 值
k_values = np.linspace(-10, 10, 100)
b_values = np.linspace(-10, 10, 100)
error_rate_table = []

for k in k_values:
    for b in b_values:
        train_errors = calculate_errors(train_features, train_labels, k, b)
        test_errors = calculate_errors(test_features, test_labels, k, b)
        error_rate_table.append((k, b, train_errors, test_errors))

# 转换为 DataFrame 并显示
error_rate_df = pd.DataFrame(error_rate_table, columns=['k', 'b', 'Training Errors', 'Testing Errors'])
optimal_threshold = error_rate_df.loc[error_rate_df['Testing Errors'].idxmin()]

print("Error Rate Table:")
print(error_rate_df)
print("\nOptimal Threshold:")
print(optimal_threshold)

# 绘制分类器
plt.figure(figsize=(10, 6))
for label, color, marker in zip(['Iris-versicolor', 'Iris-virginica'], ['blue', 'red'], ['o', 'x']):
    subset = train_data[train_data['Species'] == label]
    plt.scatter(subset['PetalLengthCm'], subset['SepalWidthCm'], c=color, label=label, marker=marker)

# 绘制最佳分类直线
k_optimal = optimal_threshold['k']
b_optimal = optimal_threshold['b']
x_values = np.linspace(features['PetalLengthCm'].min(), features['PetalLengthCm'].max(), 100)
y_values = linear_model(x_values, k_optimal, b_optimal)
plt.plot(x_values, y_values, color='green', linestyle='--', label=f'Optimal Classifier: y = {k_optimal}x + {b_optimal}')

plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.title('Optimal Linear Classifier')
plt.legend()
plt.show()


