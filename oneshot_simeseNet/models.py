# -*- coding:utf-8 -*-
'''
Create time: 2020/11/18 14:59
@Author: 大丫头
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# Different network structures, the commented out are the different experimenting structures
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Koch et al.
        # Conv2d(input_channels, output_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fcOut = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

        # VGG16
        # # dataiter = iter(train_loader)
        # # img1, img2, label = dataiter.next()
        # # print(img1.shape)
        # self.conv11 = nn.Conv2d(1, 64, 3)
        # self.conv12 = nn.Conv2d(64, 64, 3)
        # self.conv21 = nn.Conv2d(64, 128, 3)
        # self.conv22 = nn.Conv2d(128, 128, 3)
        # self.conv31 = nn.Conv2d(128, 256, 3)
        # self.conv32 = nn.Conv2d(256, 256, 3)
        # self.conv33 = nn.Conv2d(256, 256, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(256 * 8 * 8, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fcOut = nn.Linear(4096, 1)
        # self.sigmoid = nn.Sigmoid()
        # # x = self.conv11(img1)
        # # x = self.conv12(x)
        # # x = self.pool(x)
        # # x = self.conv21(x)
        # # x = self.conv22(x)
        # # x = self.pool(x)
        # # x = self.conv31(x)
        # # x = self.conv32(x)
        # # x = self.conv33(x)
        # # x = self.pool(x)
        # # print(x.shape)

    def convs(self, x):
        # Koch et al.
        # out_dim = in_dim - kernel_size + 1
        # 1, 105, 105
        x = F.relu(self.bn1(self.conv1(x)))
        # 64, 96, 96
        x = F.max_pool2d(x, (2, 2))
        # 64, 48, 48
        x = F.relu(self.bn2(self.conv2(x)))
        # 128, 42, 42
        x = F.max_pool2d(x, (2, 2))
        # 128, 21, 21
        x = F.relu(self.bn3(self.conv3(x)))
        # 128, 18, 18
        x = F.max_pool2d(x, (2, 2))
        # 128, 9, 9
        x = F.relu(self.bn4(self.conv4(x)))
        # 256, 6, 6
        return x

        # VGG16
        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.max_pool2d(x, (2,2))
        # x = F.relu(self.conv21(x))
        # x = F.relu(self.conv22(x))
        # x = F.max_pool2d(x, (2,2))
        # x = F.relu(self.conv31(x))
        # x = F.relu(self.conv32(x))
        # x = F.relu(self.conv33(x))
        # x = F.max_pool2d(x, (2,2))
        # return x

    def forward(self, x1, x2):
        x1 = self.convs(x1)

        # Koch et al.
        x1 = x1.view(-1, 256 * 6 * 6)
        x1 = self.sigmoid(self.fc1(x1))

        # VGG16
        # x1 = x1.view(-1, 256 * 8 * 8)
        # x1 = self.fc1(x1)
        # x1 = self.sigmoid(self.fc2(x1))

        x2 = self.convs(x2)

        # Koch et al.
        x2 = x2.view(-1, 256 * 6 * 6)
        x2 = self.sigmoid(self.fc1(x2))

        # VGG16
        # x2 = x2.view(-1, 256 * 8 * 8)
        # x2 = self.fc1(x2)
        # x2 = self.sigmoid(self.fc2(x2))

        x = torch.abs(x1 - x2)
        x = self.fcOut(x)
        return x
