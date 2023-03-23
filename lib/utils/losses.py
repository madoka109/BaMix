import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, reduction = 'mean'):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1) # dim:1
        index_float = index.type(torch.cuda.FloatTensor)
        # add
        self.m_list = self.m_list
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))

        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight, reduction = reduction)


class BaMixLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, p = 0, hp = 1.0, weight=None, s=30):
        super(BaMixLoss, self).__init__()
        # ldam
        # m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)) # n_j^1/4
        # m_list = m_list * (max_m / np.max(m_list)) # * C

        # p_ij = (np.array(cls_num_list).repeat(len(cls_num_list)).reshape(len(cls_num_list), -1) * p).sum(axis=0) / ((np.sum(cls_num_list) * np.diag(p)) + 1)
        p_ij = (np.array(cls_num_list).repeat(len(cls_num_list)).reshape(len(cls_num_list), -1) * p).sum(axis=0) / ((np.sum(cls_num_list) * np.diag(p)) + 1)
        
        m_list = cls_num_list / np.sum(cls_num_list) + p_ij * hp
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target, l = 0.5, reduction='mean'):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)  # dim:1

        index_float = index.type(torch.cuda.FloatTensor)

        self.m_list = ((self.m_list ) * ((1 - l) / l) + 1).log()

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))

        batch_m = batch_m.view((-1, 1))

        x_m = x - batch_m # batch_m : margin

        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s * output, target, weight=self.weight, reduction=reduction)





