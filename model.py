import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual

        return F.relu(out)
    
class EchelonNet(nn.Module):
    def __init__(self, in_channels = 13, num_res_blocks = 9, num_filters = 256):
        super(EchelonNet, self).__init__()

        # Initialized convolution
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size = 3, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Residual tower (9 blocks * 2 layers, 1 initial + 1 head = ~20 layers)
        self.res_tower = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        # Value Head (board evaluation)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size = 1)
        self.value_bn = nn.BatchNorm2d(num_filters)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_tower(x)

        # Value head logic
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.value_fc1(v))        
        v = torch.tanh(self.value_fc2(v)) # scale the value between -1 and 1

        return v
    
