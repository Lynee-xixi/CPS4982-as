import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载MNIST数据集
(x, y), (x_test, y_test) = mnist.load_data()

# 将数据集分割成训练集和验证集（80%训练集，20%验证集）
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 数据转换和归一化
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
x_val = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1) / 255.0
y_val = torch.tensor(y_val, dtype=torch.long)
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(y_test, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()

# 定义不同学习率
learning_rates = [0.01, 0.001, 0.0001]
num_epochs = 5
accuracy_results = {}

for lr in learning_rates:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 计算验证集准确率
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(
            f'Learning Rate: {lr}, Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
            f'Validation Accuracy: {val_accuracy:.2f}%')

    accuracy_results[lr] = val_accuracies

# 绘制不同学习率下的验证集准确率
for lr, accuracies in accuracy_results.items():
    plt.plot(range(1, num_epochs + 1), accuracies, label=f'LR={lr}')

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy vs. Epoch for Different Learning Rates')
plt.show()


# 评估模型在测试集上的准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')

# 显示一个错误分类的示例
model.eval()
misclassified_found = False
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                img = images[i].squeeze().numpy()
                plt.imshow(img, cmap='gray')
                plt.title(f'True label: {labels[i]}, Predicted: {predicted[i]}')
                plt.show()
                print(f'True label: {labels[i]}, Predicted: {predicted[i]}')
                misclassified_found = True
                break
        if misclassified_found:
            break
