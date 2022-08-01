import torch
import torch.nn as nn

from .pose_decoder import PoseDecoder
from .resnet_encoder import ResnetEncoder

from project_3d import intrinsic_vec2mat, pose_vec2mat, inverse_pose

class PoseBase(nn.Module):
    def __init__(self, opt):
        super(PoseBase, self).__init__()
        self.opt = opt
        self.register_buffer('head', torch.zeros(1, 4, 4))

    def get_intrinsics(self, out, width, height):
        f = out.detach() * 0 + self.opt.camera_scale 
        cx, cy = width * 0.5, height * 0.5
        K, K_inv = intrinsic_vec2mat(f, cx, cy)
        return K, K_inv

    def forward(self, images, fixed=False):
        pass

    def save_weight(self, path):
        pass

    def load_weight(self, path):
        pass
