import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter

__all__ = ['ResNet', 'resnet10i', 'resnet50i']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=None, use_norm=False):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.use_norm = use_norm

        self.conv0s = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0s = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1s = self._make_layer(block, 64, layers[0], stride=1)
        self.in_planes = self.next_in_planes
        self.layer2s =self._make_layer(block, 128, layers[1], stride=2)
        self.in_planes = self.next_in_planes
        self.layer3s = self._make_layer(block, 256, layers[2], stride=2)
        self.in_planes = self.next_in_planes
        self.layer4s = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.in_planes = self.next_in_planes
        if use_norm:
            self.linears = NormedLinear(512 * block.expansion, self.num_classes)
        else:
            self.linears = nn.Linear(512 * block.expansion, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        self.next_in_planes = self.in_planes
        if stride != 1 or self.next_in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.next_in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.next_in_planes, planes, stride, downsample))
        self.next_in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.next_in_planes, planes))

        return nn.Sequential(*layers)
    def _getfeature(self, x):
        out = (self.conv0s)(x)
        out = (self.bn0s)(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = (self.layer1s)(out)
        out = (self.layer2s)(out)
        out = (self.layer3s)(out)
        out = (self.layer4s)(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def _getoutput(self, x):

        out = (self.linears)(x)
        return out

    def forward(self, x, feature_flag=False, train = False):
        out = (self.conv0s)(x)
        out = (self.bn0s)(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = (self.layer1s)(out)
        out = (self.layer2s)(out)
        out = (self.layer3s)(out)
        out = (self.layer4s)(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        if feature_flag == True:
            return out
        out = (self.linears)(out)

        return out


def resnet10i(num_classes=None, use_norm=False):
    print('Loading Scratch ResNet 10 Feature Model.')
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, use_norm=use_norm)


def resnet50i(num_classes=None, use_norm=False):
    print('Loading Scratch ResNet 50 Feature Model.')
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, use_norm=use_norm)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
