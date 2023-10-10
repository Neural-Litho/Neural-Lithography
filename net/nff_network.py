

""" from neural photo finishing
"""

import torch.nn as nn 

class NeuralPointwiseNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, ch=64, leak = 0.2):
        super(NeuralPointwiseNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=2*ch, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=2*ch, out_channels=ch, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1)
        self.lrelu0 = nn.LeakyReLU(leak)  # was 0.2, 0.01 for pbl3d1 saved
        self.lrelu1 = nn.LeakyReLU(leak)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        return x
    

class NeuralAreawiseNet(nn.Module):
    """Some Information about MyModule"""

    def __init__(self, ch=32, leak=0.2):
        super(NeuralAreawiseNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=2*ch, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels=2*ch, out_channels=2*ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2*ch, out_channels=ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
        in_channels=ch, out_channels=ch, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
        in_channels=ch, out_channels=1, kernel_size=3, padding=1)
        
        self.lrelu0 = nn.LeakyReLU(leak)  # was 0.2, 0.01
        self.lrelu1 = nn.LeakyReLU(leak)
        self.lrelu2 = nn.LeakyReLU(leak)
        self.lrelu3 = nn.LeakyReLU(leak)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lrelu0(x)
        x = self.conv1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.lrelu3(x)
        x = self.conv4(x)
        return x
