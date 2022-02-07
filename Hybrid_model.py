import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolutional part from netfreq1dkernel5env_output for generating feature representations


class NetConv(nn.Module):

    def __init__(self, C=19):
        super(NetConv, self).__init__()

        n1 = 8
        n2 = 8
        dil = 1

        self.conv1 = nn.Conv2d(C, n1, (5, 10), dilation=(1, dil))
        self.conv2 = nn.Conv2d(n1, n2, (5, 5), dilation=(1, dil))
        self.bn1 = nn.BatchNorm2d(n1)
        self.bn2 = nn.BatchNorm2d(n2)

        self.conv1b = nn.Conv2d(C, n1, (5, 5), dilation=(1, dil))
        self.conv2b = nn.Conv2d(n1, n2, (5, 10), dilation=(1, dil))
        self.bn1b = nn.BatchNorm2d(n1)
        self.bn2b = nn.BatchNorm2d(n2)

        self.conv1c = nn.Conv2d(C, n1, (5, 1), dilation=(1, dil))
        self.conv2c = nn.Conv2d(n1, n2, (5, 10), dilation=(1,2*dil))
        self.bn1c = nn.BatchNorm2d(n1)
        self.bn2c = nn.BatchNorm2d(n2)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.bn1(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.bn2(x1)

        x2 = self.conv1b(x)
        x2 = F.relu(x2)
        x2 = self.bn1b(x2)
        x2 = self.conv2b(x2)
        x2 = F.relu(x2)
        x2 = self.bn2b(x2)

        x3 = self.conv1c(x)
        x3 = F.relu(x3)
        x3 = self.bn1c(x3)
        x3 = self.conv2c(x3)
        x3 = F.relu(x3)
        x3 = self.bn2c(x3)

        N, C1, _, _ = x1.shape
        x1 = F.adaptive_avg_pool2d(x1, 1).view(N, C1)

        N, C2, _, _ = x2.shape
        x2 = F.adaptive_avg_pool2d(x2, 1).view(N, C2)

        N, C3, _, _ = x3.shape
        x3 = F.adaptive_avg_pool2d(x3, 1).view(N, C3)

        o1 = torch.cat((x1, x2, x3), dim=1)

        return o1


class NetIndividual(nn.Module):

    def __init__(self, C=19):
        super(NetIndividual, self).__init__()
        self.conv = NetConv(C=C)
        self.lin = nn.Linear(24, 1)

    def set_logreg(self, coef, intercept):
        with torch.no_grad():
            self.lin.bias.data[:] = torch.from_numpy(intercept)
            self.lin.weight.data[:] = torch.from_numpy(coef)

    def forward(self, x):
        return torch.sigmoid(self.lin(self.conv(x)))