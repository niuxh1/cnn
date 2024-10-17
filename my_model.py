import torch.nn as nn
import torch.nn.functional as F
import torch


class model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.cnn1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.BN = nn.BatchNorm2d(128)
        self.ReLU = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.size0 = input_size // 2
        self.cnn2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.BN2 = nn.BatchNorm2d(128)
        self.ReLU2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.size1 = self.size0 // 2
        self.fc1 = nn.Linear(self.size1 * self.size1 * 128, hidden_size)
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.BN(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.cnn2(x)
        x = self.BN2(x)
        x = self.ReLU2(x)
        x = self.maxpool2(x)
        x=x.view(-1, self.size1 * self.size1 * 128)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = F.softmax(x, dim=1)
        return x
