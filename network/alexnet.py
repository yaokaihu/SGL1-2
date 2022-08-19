import torch.nn as nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1)

        self.bn1 = nn.BatchNorm2d(96)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)

        self.bn2 = nn.BatchNorm2d(256)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1)

        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1)

        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1)

        self.bn5 = nn.BatchNorm2d(256)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.linear1 = nn.Linear(1024, 2048)

        self.dropout1 = nn.Dropout(0.5)

        self.linear2 = nn.Linear(2048, 2048)

        self.dropout2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(2048, 200)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = F.relu(self.bn3(self.conv3(out)))

        out = F.relu(self.bn4(self.conv4(out)))

        out = F.relu(self.bn5(self.conv5(out)))

        out = self.pool3(out)

        out = out.reshape(-1, 256 * 2 * 2)

        out = F.relu(self.linear1(out))

        out = self.dropout1(out)

        out = F.relu(self.linear2(out))

        out = self.dropout2(out)

        out = self.linear3(out)

        return out
