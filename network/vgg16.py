import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        # GROUP 1
        self.conv1_1 = nn.Conv2d(3, 64, 3,
                                 padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3,
                                 padding=1)

        self.bn1_2 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        # GROUP 2
        self.conv2_1 = nn.Conv2d(64, 128, 3,
                                 padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3,
                                 padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # GROUP 3
        self.conv3_1 = nn.Conv2d(128, 256, 3,
                                 padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3,
                                 padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        # GROUP 4
        self.conv4_1 = nn.Conv2d(256, 512, 3,
                                 padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3,
                                 padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        # GROUP 5
        self.conv5_1 = nn.Conv2d(512, 512, 3,
                                 padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3,
                                 padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 16
            nn.Linear(4096, num_classes),
        )

    # 定义前向传播
    def forward(self, x):
        input_dimen = x.size(0)

        # GROUP 1
        output = self.conv1_1(x)
        output = self.bn1_1(output)
        output = F.relu(output)
        output = self.conv1_2(output)
        output = self.bn1_2(output)
        output = F.relu(output)
        output = self.maxpool1(output)

        # GROUP 2
        output = self.conv2_1(output)
        output = self.bn2_1(output)
        output = F.relu(output)
        output = self.conv2_2(output)
        output = self.bn2_2(output)
        output = F.relu(output)
        output = self.maxpool2(output)

        # GROUP 3
        output = self.conv3_1(output)
        output = self.bn3_1(output)
        output = F.relu(output)
        output = self.conv3_2(output)
        output = self.bn3_2(output)
        output = F.relu(output)
        output = self.conv3_3(output)
        output = self.bn3_3(output)
        output = F.relu(output)
        output = self.maxpool3(output)

        # GROUP 4
        output = self.conv4_1(output)
        output = self.bn4_1(output)
        output = F.relu(output)
        output = self.conv4_2(output)
        output = self.bn4_2(output)
        output = F.relu(output)
        output = self.conv4_3(output)
        output = self.bn4_3(output)
        output = F.relu(output)
        output = self.maxpool4(output)

        # GROUP 5
        output = self.conv5_1(output)
        output = self.bn5_1(output)
        output = F.relu(output)
        output = self.conv5_2(output)
        output = self.bn5_2(output)
        output = F.relu(output)
        output = self.conv5_3(output)
        output = self.bn5_3(output)
        output = F.relu(output)
        output = self.maxpool5(output)
        output = self.avg(output)

        output = output.view(x.size(0), -1)

        output = self.classifier(output)

        return output


def vgg16(num_classes):
    return VGG16(num_classes)
