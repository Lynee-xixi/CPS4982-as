import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 计算均值的函数
def calculate_mean(data):
    return np.sum(data, axis=0) / data.shape[0]


# 计算标准差的函数
def calculate_std(data, mean):
    return np.sqrt(np.sum((data - mean) ** 2, axis=0) / data.shape[0])


# 读取数据
iris_path = 'IrisData.csv'
iris_data = pd.read_csv(iris_path)
# 筛选出Versicolor和Virginica的数据
iris_filtered = iris_data[(iris_data['Species'] == 'Iris-versicolor') | (iris_data['Species'] == 'Iris-virginica')]
# 选择特定的特征
features = iris_filtered[['PetalLengthCm', 'SepalWidthCm']]
labels = iris_filtered['Species'].apply(lambda x: 1 if x == 'Iris-versicolor' else -1)

# 将数据分为训练集和测试集
X_train = features.iloc[:80].values  # 使用每个类的前40个样本作为训练集
X_test = features.iloc[80:].values   # 使用每个类的剩余10个样本作为测试集
y_train = labels.iloc[:80].values
y_test = labels.iloc[80:].values
# 计算均值和标准差以进行归一化
mean_train = calculate_mean(X_train)
std_train = calculate_std(X_train, mean_train)
# 应用零均值归一化
X_train_normalized = (X_train - mean_train) / std_train
X_test_normalized = (X_test - mean_train) / std_train

print("Training features normalized:\n", X_train_normalized)
print("Testing features normalized:\n", X_test_normalized)
print("y_train:\n", y_train)
print("y_test:\n", y_test)


# 使用正态分布随机初始化权重和偏置
weights = np.random.normal(0, 1, 2)  # 两个输入特征，因此权重有两个
bias = np.random.normal(0, 1)  # 单个偏置

# 打印初始化的权重和偏置
print("Initialized weights:", weights)
print("Initialized bias:", bias)


# 定义感知机的激活函数
def perceptron_activation(sigma):
    return 1 if sigma >= 0 else -1


# 输入样本，花瓣长度4.7厘米，萼片宽度2.4厘米
sample = np.array([4.7, 2.4])

# 计算加权和
sigma = np.dot(weights, sample) + bias

# 使用激活函数计算输出
output = perceptron_activation(sigma)

# 打印输出
print("Output of the perceptron for the input sample:", output)


# 数据和标签
X_train = X_train_normalized
y_train = y_train
X_test = X_test_normalized
y_test = y_test


# 激活函数
def perceptron_activation(sigma):
    return 1 if sigma >= 0 else -1


# 准确率计算函数
def calculate_accuracy(X, y, weights, bias):
    correct_predictions = 0
    for i in range(X.shape[0]):
        sigma = np.dot(weights, X[i]) + bias
        prediction = perceptron_activation(sigma)
        if prediction == y[i]:
            correct_predictions += 1
    return correct_predictions / len(X)


# 感知机训练及验证函数
def train_and_validate_perceptron(X_train, y_train, X_test, y_test, epochs, learning_rate):
    weights = np.random.normal(0, 1, X_train.shape[1])
    bias = np.random.normal(0, 1)
    training_accuracy = []
    testing_accuracy = []

    for epoch in range(epochs):
        # 训练过程
        for i in range(X_train.shape[0]):
            sigma = np.dot(weights, X_train[i]) + bias
            prediction = perceptron_activation(sigma)
            error = y_train[i] - prediction  # 计算误差
            weights += learning_rate * error * X_train[i]  # 更新权重
            bias += learning_rate * error  # 更新偏置

        # 计算训练集和测试集的准确率
        train_acc = calculate_accuracy(X_train, y_train, weights, bias)
        test_acc = calculate_accuracy(X_test, y_test, weights, bias)
        training_accuracy.append(train_acc)
        testing_accuracy.append(test_acc)
        print(f"Epoch {epoch + 1}: Train Accuracy = {train_acc}, Test Accuracy = {test_acc}")

    return weights, bias, training_accuracy, testing_accuracy


# 设置学习率和迭代次数
learning_rates = [0.003, 0.01, 0.03]
epochs = 5
results = []

# 对每个学习率进行训练和验证
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    _, _, train_acc, test_acc = train_and_validate_perceptron(X_train, y_train, X_test, y_test, epochs, lr)
    results.append(pd.DataFrame({
        'Epoch': range(1, epochs + 1),
        'Training Accuracy': train_acc,
        'Testing Accuracy': test_acc
    }).set_index('Epoch'))
    plt.plot(range(1, epochs+1), train_acc, marker='o', label=f'Training lr={lr}')
    plt.plot(range(1, epochs+1), test_acc, marker='x', label=f'Testing lr={lr}')

# 显示图表
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 显示表格
for idx, df in enumerate(results):
    print(f"Learning Rate: {learning_rates[idx]}")
    print(df, "\n")
