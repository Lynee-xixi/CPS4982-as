import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


# 定义数据处理函数
def calculate_mean(data):
    return np.sum(data, axis=0) / data.shape[0]


def calculate_std(data, mean):
    return np.sqrt(np.sum((data - mean) ** 2, axis=0) / data.shape[0])


# 读取和处理数据
iris_path = 'IrisData.csv'
iris_data = pd.read_csv(iris_path)
iris_filtered = iris_data[(iris_data['Species'] == 'Iris-versicolor') | (iris_data['Species'] == 'Iris-virginica')]
features = iris_filtered[['PetalLengthCm', 'SepalWidthCm']]
labels = iris_filtered['Species'].apply(lambda x: 0 if x == 'Iris-versicolor' else 1).values

# 划分数据集
versicolor_indices = np.where(labels == 0)[0]
virginica_indices = np.where(labels == 1)[0]

X_train = np.concatenate((features.iloc[versicolor_indices[:40]].values, features.iloc[virginica_indices[:40]].values), axis=0)
X_test = np.concatenate((features.iloc[versicolor_indices[40:50]].values, features.iloc[virginica_indices[40:50]].values), axis=0)
y_train = np.concatenate((labels[versicolor_indices[:40]], labels[virginica_indices[:40]]), axis=0)
y_test = np.concatenate((labels[versicolor_indices[40:50]], labels[virginica_indices[40:50]]), axis=0)


mean_train = calculate_mean(X_train)
std_train = calculate_std(X_train, mean_train)
X_train_normalized = (X_train - mean_train) / std_train
X_test_normalized = (X_test - mean_train) / std_train

print("Training features normalized:\n", X_train_normalized)
print("Testing features normalized:\n", X_test_normalized)
print("y_train:\n", y_train)
print("y_test:\n", y_test)


# 定义多层感知机模型
class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10)

model = MultiLayerPerceptron()
print(model)


# 训练函数
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# 测试函数
def test(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    accuracy = 100 * correct / total
    return accuracy


# 主训练循环
learning_rates = [0.01, 0.05, 0.1]
all_train_losses = {}
all_train_accuracies = {}
all_test_accuracies = {}

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    model = MultiLayerPerceptron()  # 模型初始化
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 训练和评估
    for epoch in range(5):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion)
        test_accuracy = test(model, test_loader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

    all_train_losses[lr] = train_losses
    all_train_accuracies[lr] = train_accuracies
    all_test_accuracies[lr] = test_accuracies

# 结果可视化
plt.figure(figsize=(10, 5))
for lr in learning_rates:
    plt.plot(all_train_accuracies[lr], label=f'Train Acc LR={lr}')
    plt.plot(all_test_accuracies[lr], label=f'Test Acc LR={lr}', linestyle='--')
plt.title('Training and Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.show()


