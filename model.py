# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.functional as F
class IN_Relu(nn.Module):
    def __init__(self, in_channels):
        super(IN_Relu, self).__init__()
        self.IN = nn.InstanceNorm2d(in_channels)
    def forward(self, X):
        Y = F.relu(self.IN(X));
        return Y

class trans_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(trans_conv, self).__init__()
        self.name = 'trans_conv'
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1)
    def forward(self, X):
        return self.conv(X)

class classify_conv(nn.Module):
    def __init__(self, in_channels, num_cls, k=1):
        super(classify_conv, self).__init__()
        self.name = 'classify_conv'
        p = int((k-1)/2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=num_cls, kernel_size=(k, k), padding=(p, p), stride=1)

    def forward(self, X):
        return self.conv(X)
class feat_extract_block(nn.Module):
    def __init__(self, in_channels, out_channels, k=1):
        super(feat_extract_block, self).__init__()
        self.name = 'feat_extract_block'
        p = int((k-1)/2)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=k, padding=p, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, padding=p, stride=1)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        return X

class TestNet(nn.Module):
    def __init__(self, in_channels, num_cls):
        super(TestNet, self).__init__()
        self.layer_channels = 16
        self.layer_num = 4
        self.mid_in_channels = 30
        self.mid_out_channels = self.layer_num * self.layer_channels
        self.dilate_rate = range(1, self.layer_num + 1)
        # self.dilate_rate = [1, 1, 1]
        self.first_conv = trans_conv(in_channels, self.mid_in_channels)
        self.feat_extract_block = feat_extract_block(self.mid_in_channels, self.mid_out_channels)
        self.classify_conv = classify_conv(self.mid_out_channels, num_cls, k=3)
        self.name = 'TestNet'

    def forward(self, X):
        X = self.first_conv(X)
        out = self.feat_extract_block(X)
        out = self.classify_conv(out)
        return out#, prob




