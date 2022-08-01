# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import math
import torch
import torch.nn as nn
from collections import OrderedDict

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, indices: list) -> torch.Tensor:
        pe = self.pe[indices].view(-1, self.d_model, 1, 1).expand_as(x)
        x = x + pe.detach()
        return x #self.dropout(x)

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, intrinsic=False, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc

        self.output_ch = 7 if intrinsic else 6

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, stride, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, self.output_ch, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        out = self.relu(self.convs['squeeze'](input_features))

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
        
        out = out.mean(3).mean(2)

        out = 1e-2 * out.view(-1, self.output_ch)
         
        return out

