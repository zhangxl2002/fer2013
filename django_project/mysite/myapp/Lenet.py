import torch
import torch.nn as nn
# a=16
# b=32

# 定义LeNet模型
class LeNet(nn.Module):
    a=16
    b=32
    c=120
    d=64
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, a, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(a, b, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(b * 9 * 9, c)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(c, d)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(d, 7)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, b * 9 * 9)
        x = self.dropout(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.relu4(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x