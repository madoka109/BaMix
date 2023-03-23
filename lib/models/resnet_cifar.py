'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
import numpy as np

__all__ = ['Classifier','ResNet_s', 'resnet10', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=None, use_norm=False):
        super(ResNet_s, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 16
        self.conv0s = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0s = nn.BatchNorm2d(16) 
        self.layer1s = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes
        self.layer2s = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.in_planes = self.next_in_planes
        self.layer3s = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.in_planes = self.next_in_planes
        if use_norm:
            self.linears = NormedLinear(64, self.num_classes)
        else:
            self.linears = nn.Linear(64, self.num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _getfeature(self, x):
        out = (self.conv0s)(x)
        out = (self.bn0s)(out)
        out = F.relu(out)
        out = (self.layer1s)(out)
        out = (self.layer2s)(out)
        out = (self.layer3s)(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

    def _getoutput(self, x):
        out = (self.linears)(x)
        return out

    def forward(self, x, feature_flag=False, train = False):
        # train
        if train:
            if feature_flag == True:
                feature = self._getfeature(x)
                return feature
            else:
                feature = self._getfeature(x)
                out = self._getoutput(feature)
                return out
        # test
        else:
            feature = self._getfeature(x)
            out = self._getoutput(feature)
            if(feature_flag == False):
                return out
            else:
                return feature

class ResNet_c(nn.Module):

    def __init__(self, block, num_blocks, num_classes=None, use_norm=False, num_experts=None):
        super(ResNet_s, self).__init__()
        self.num_classes = num_classes
        self.in_planes = 16
        self.num_experts = num_experts
        # self.conv0s = nn.ModuleList(
        #     [nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) for _ in range(self.num_experts)])
        # self.bn0s = nn.ModuleList(
        #     [nn.BatchNorm2d(16) for _ in range(self.num_experts)])
        # self.layer1s = nn.ModuleList(
        #     [self._make_layer(block, 16, num_blocks[0], stride=1) for _ in range(self.num_experts)])
        # self.in_planes = self.next_in_planes
        # self.layer2s = nn.ModuleList(
        #     [self._make_layer(block, 32, num_blocks[1], stride=2) for _ in range(self.num_experts)])
        # self.in_planes = self.next_in_planes
        # self.layer3s = nn.ModuleList(
        #     [self._make_layer(block, 64, num_blocks[2], stride=2) for _ in range(self.num_experts)])
        self.conv0s = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0s = nn.BatchNorm2d(16)
        self.layer1s = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes
        self.layer2s = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.in_planes = self.next_in_planes
        self.layer3s = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.in_planes = self.next_in_planes
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(64, self.num_classes) for _ in range(self.num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(64, self.num_classes) for _ in range(self.num_experts)])
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _getfeature(self, x, expert = None):
        # out = (self.conv0s[expert])(x)
        # out = (self.bn0s[expert])(out)
        # out = F.relu(out)
        # out = (self.layer1s[expert])(out)
        # out = (self.layer2s[expert])(out)
        # out = (self.layer3s[expert])(out)
        out = (self.conv0s)(x)
        out = (self.bn0s)(out)
        out = F.relu(out)
        out = (self.layer1s)(out)
        out = (self.layer2s)(out)
        out = (self.layer3s)(out)

        return out

    def _getoutput(self, x, expert = None):
        out = F.avg_pool2d(x, x.size()[3])
        out = out.view(out.size(0), -1)
        out = (self.linears[expert])(out)
        return out

    def forward(self, x, expert=None, feature_flag=False, train = False):
        # train
        if train:
            if feature_flag == True:
                feature = self._getfeature(x, expert)
                return feature
            else:
                feature = self._getfeature(x, expert)
                out = self._getoutput(feature, expert)
                return out
        # test
        else:
            feature = self._getfeature(x, expert)
            out = self._getoutput(feature, expert)
            if(feature_flag == False):
                return out
            else:
                return feature

class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, num_experts = None):
        super(Classifier, self).__init__()
        self.num_experts = num_experts

        self.fc = nn.ModuleList([nn.Linear(feat_in, num_classes) for _ in range(self.num_experts)])
        self.apply(_weights_init)

    def forward(self, x, expert = None):
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = (self.fc[expert])(x)
        return x

def resnet10(num_classes=10, use_norm=False, num_experts=2):
    return ResNet_s(BasicBlock, [1, 1, 1], num_classes=num_classes, use_norm=use_norm, num_experts=num_experts)


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=None, use_norm=False, num_experts=None):
    print('Loading Scratch ResNet 32 Feature Model.')
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


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