import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import my_model  # 你的模型文件

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试集
test_dataset = datasets.CIFAR100(root='./data', download=True, train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# 加载模型
model = my_model.model(input_size=32, hidden_size=256, output_size=100).to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置模型为评估模式

# 测试模型
correct = 0
total = 0

with torch.no_grad():  # 在测试中不需要计算梯度
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
