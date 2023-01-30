import vdp
import torch
import torch.nn as nn
import pytorch_lightning as pl


class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.conv1 = vdp.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = vdp.BatchNorm2d(out_channels)
        self.conv2 = vdp.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = vdp.BatchNorm2d(out_channels)
        self.conv3 = vdp.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = vdp.BatchNorm2d(out_channels * 4)
        self.relu = vdp.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        mu, sigma = x[0], x[1]
        identity = x

        mu, sigma = self.conv1(mu, sigma)
        mu, sigma = self.bn1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.conv2(mu, sigma)
        mu, sigma = self.bn2(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.conv3(mu, sigma)
        mu, sigma = self.bn3(mu, sigma)

        # Downsample only if the number of channels mismatch
        if self.identity_downsample is not None:
            # print('HEYYYY LISTEN!!')
            identity = self.identity_downsample(identity)

        # Skip connection
        mu += identity[0]
        sigma = (torch.eye(sigma.shape[2]).to('cuda')**2) * identity[1]

        mu, sigma = self.relu(mu, sigma)

        return {0: mu, 1: sigma}


class ResNet(pl.LightningModule):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = vdp.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, input_flag=True)
        self.bn1 = vdp.BatchNorm2d(64)
        self.relu = vdp.ReLU()
        self.maxpool = vdp.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        # self.avgpool = vdp.AdaptiveAvgPool2d((1, 1))
        self.fc = vdp.Linear(2048, num_classes)
        self.softmax = vdp.Softmax()
        # self.final_pool = vdp.MaxPool2d(kernel_size=6, stride=4, padding=1) #Cifar100
        self.final_pool = vdp.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        mu, sigma = self.conv1(x)
        mu, sigma = self.bn1(mu, sigma)
        mu, sigma = self.relu(mu, sigma)
        mu, sigma = self.maxpool(mu, sigma)

        # nn.Sequential can only take 1 input in forward so work around is put mu and sigma in a dict and unpack it in the forward
        x = {0: mu, 1: sigma}

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        mu, sigma = x[0], x[1]

        mu, sigma = self.final_pool(mu, sigma)
        mu = mu.reshape(mu.shape[0], -1)
        sigma = sigma.reshape(sigma.shape[0], -1)
        mu, sigma = self.fc(mu, sigma)
        mu, sigma = self.softmax(mu, sigma)

        return mu, sigma

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(IdentityBlock(self.in_channels, out_channels * 4, stride))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def score(self, logits, y):
        score = torch.sum(torch.argmax(logits, dim=1) == y)/len(logits)
        return score.cpu().numpy()


class IdentityBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(IdentityBlock, self).__init__()
        self.conv1 = vdp.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = vdp.BatchNorm2d(out_channels)

    def forward(self, x):
        mu, sigma = x[0], x[1]
        mu, sigma = self.conv1(mu, sigma)
        mu, sigma = self.bn(mu, sigma)

        return {0: mu, 1: sigma}