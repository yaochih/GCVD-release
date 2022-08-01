import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

def deactivate_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

class DepthBase(nn.Module):
    def __init__(self, opt, seq_io):
        super(DepthBase, self).__init__()

        self.opt = opt
        self.seq_io = seq_io

    def forward(self, images, normalize=True):
        pass

    def save_weight(self, path):
        pass

    def load_weight(self, path):
        pass

    def merge_weight(self, path):
        pass

    def freeze(self):
        pass
