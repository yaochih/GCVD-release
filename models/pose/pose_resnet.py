import torch
import torch.nn as nn

from .pose_decoder import PoseDecoder, PositionalEncoding
from .resnet_encoder import ResnetEncoder

from .pose_base import PoseBase

from project_3d import intrinsic_vec2mat, pose_vec2mat, inverse_pose

def deactivate_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

class PoseResnet(PoseBase):
    def __init__(self, opt, seq_io):
        super(PoseResnet, self).__init__(opt)
        
        dim_encoding = 512
        self.encoder = ResnetEncoder(18, True)
        self.decoder = PoseDecoder([dim_encoding], opt.learn_intrinsic)
        self.enable_pe = opt.positional_encoding        
        if self.enable_pe:
            self.pe = PositionalEncoding(dim_encoding, max_len=seq_io.length)

        self.train()

    def train(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.apply(deactivate_bn)

        for p in self.decoder.parameters():
            p.requires_grad = True

    def set_feature(self, images, indices=None):
        #return
        self.encoding = self.encoder(images)
        if self.enable_pe and indices is not None:
            self.encoding = self.pe(self.encoding, indices)

    def forward(self, images, indices=None):
        shape = images.shape
        #self.encoding = self.encoder(images)
        #if self.positional_encoding and indices is not None:
        #    self.encoding = self.pe(self.encoding, indices)
        out = self.decoder(self.encoding)

        pose_vec = out[:, :6]

        pose = pose_vec2mat(pose_vec)
        
        if self.opt.learn_intrinsic: 
            K, K_inv = self.get_intrinsics(out[:, -1:], shape[-1], shape[-2])
        else:
            K, K_inv = self.get_intrinsics(out[:, -1:].detach() * 0, shape[-1], shape[-2])
        
        return pose, inverse_pose(pose), K, K_inv

    def save_weight(self, path):
        torch.save(self.state_dict(), path)

    def load_weight(self, path):
        self.load_state_dict(torch.load(path))

    def merge_weight(self, path):
        new_weights = torch.load(path)
        own_state = self.state_dict()
        for name, param in new_weights.items():
            own_state[name].copy_((own_state[name] + param) * 0.5)
