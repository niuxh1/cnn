import torch
import torchvision.transforms as transforms

import my_model
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

dataset = {}

dataset["train"] = datasets.CIFAR100(root='./data', download=True, train=True, transform=transform)
dataset["test"] = datasets.CIFAR100(root='./data', download=True, train=False, transform=transform)

train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=32, shuffle=True)

model = my_model.model(input_size=32, hidden_size=256, output_size=100).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criteria=torch.nn.CrossEntropyLoss()

losses = []

for epoch in range(100):
    model.train()
    running_loss = 0
    print(f"Epoch {epoch}")
    accuracy = 0
    labels_size=0
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'Processed {batch_idx * train_loader.batch_size + images.size(0)} images', end='\r')
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = criteria(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        score=outputs.argmax(dim=1)
        labels_size+=labels.size(0)
        accuracy += (score == labels.to(device)).sum().item()
    loss = running_loss / len(train_loader)
    print(f"loss:{loss},accuracy:{100*accuracy / labels_size}%" )
    losses.append(loss)

torch.save(model.state_dict(), 'model.pth')

plt.figure(figsize=(10, 5))
plt.plot(range(1, 10 + 1), losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(range(1, 10 + 1))  # 设置 x 轴刻度
plt.grid()
plt.show()
