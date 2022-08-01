import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .midas.midas_net import MidasNet
from .midas.midas_net_custom import MidasNet_small
from .midas.dpt_depth import DPTDepthModel
from .depth_base import *

DEPTH_WEIGHT_PATH = 'weights/models/midas_v21-f6b98070.pt'

class DepthMidas(DepthBase):
    def __init__(self, opt, seq_io):
        super(DepthMidas, self).__init__(opt, seq_io)

        self.model = MidasNet(DEPTH_WEIGHT_PATH, non_negative=True)
        norm_mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        norm_std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        
        self.train()

        self.sigmoid = nn.Sigmoid()
        self.register_buffer('norm_mean', norm_mean)
        self.register_buffer('norm_std',  norm_std)
        
        h, w = seq_io.image_size['down']
        self.layer_norm = nn.LayerNorm((256, h // 2, w // 2), elementwise_affine=False)

    def train(self):
        for name, param in self.model.named_parameters():
            if 'output_conv.4' in name or 'output_conv.2' in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False
        self.model.apply(deactivate_bn)

    def extract_feature(self, images):
        shape = images.shape
        # [..., C, H, W] -> [X, C, H, W]
        C, H, W = shape[-3:]
        input_ = images.reshape(-1, C, H, W)
        input_ = (images - self.norm_mean) / self.norm_std
        output = self.model.extract_feature(input_)
        return output

    def set_feature(self, images):
        self.feature = self.extract_feature(images)
        self.shape = images.shape
        
    def forward(self, images):
        output = self.model(self.layer_norm(self.feature))
        output_size = output.shape[-2:]

        B = self.shape[0] 
        # [X, 1, H, W] -> [..., H, W]
        disp = output.reshape(self.shape[:-3] + output.shape[-2:])
        
        depth = (disp + 1e-7).reciprocal() 
        depth = torch.clip(depth.unsqueeze(1), 1e-3, 1) * 10

        return depth 

    def save_weight(self, path):
        weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights[name] = param
        torch.save(weights, path)

    def load_weight(self, path):
        weights = torch.load(path)
        own_state = self.model.state_dict()
        for name, param in weights.items():
            own_state[name].copy_(param)

    def merge_weight(self, path):
        new_weights = torch.load(path)
        own_state = self.model.state_dict()
        for name, param in new_weights.items():
            own_state[name].copy_((own_state[name] + param) * 0.5)

    def freeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
