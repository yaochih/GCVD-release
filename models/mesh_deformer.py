import sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_grid

class MeshDeformer(nn.Module):
    def __init__(self, opt, seq_io):
        super(MeshDeformer, self).__init__()

        h, w = seq_io.image_size['down']
        if h > w: a, b = h, w
        else:     b, a = h, w

        grid_a = opt.mesh_size
        grid_b = int(np.round(b * grid_a / a))
        if h > w: grid_h, grid_w = grid_a, grid_b
        else:     grid_w, grid_h = grid_a, grid_b
        self.grid_h, self.grid_w = grid_h + 1, grid_w + 1
    
    def dense_mesh(self, factor=2):
        self.grid_h = (self.grid_h - 1) * factor + 1
        self.grid_w = (self.grid_w - 1) * factor + 1

    def forward(self, depth):
        mesh_res = F.interpolate(torch.exp(self.mesh), depth.shape[-2:], mode='bilinear', align_corners=True)
        return depth * mesh_res

    def create_mesh(self, dyn_masks):
        mesh = torch.zeros(dyn_masks.shape[0], 1, self.grid_h, self.grid_w)
        self.mesh = nn.Parameter(mesh.to(dyn_masks.device))
        
        self.mesh.requires_grad_()
        self.train()     

    def get_weight_map(self, dyn_masks):
        # reference: https://github.com/facebookresearch/robust_cvd/blob/main/lib/PoseOptimizer.cpp#L559
        self.create_mesh(dyn_masks)
        with torch.no_grad():
            gh, gw = self.grid_h, self.grid_w 
            B, _, dh, dw = dyn_masks.shape

            static_weights = torch.zeros(B, gh, gw).to(dyn_masks.device)
            dynamic_weights = torch.zeros(B, gh, gw).to(dyn_masks.device)
           
            grid = get_grid(dyn_masks)

            fy = grid[:, 1] * (gh - 1) / dh
            iy = fy.long()
            ry = fy - iy

            fx = grid[:, 0] * (gw - 1) / dw
            ix = fx.long()
            rx = fx - ix

            w0 = (1 - rx) * (1 - ry)
            w1 = rx * (1 - ry)
            w2 = (1 - rx) * ry
            w3 = rx * ry

            s_map = (dyn_masks.squeeze(1) > 0.5).float() * 1
            d_map = (dyn_masks.squeeze(1) <= 0.5).float() * 1

            w0_s = w0 * s_map; w0_d = w0 * d_map
            w1_s = w1 * s_map; w1_d = w1 * d_map
            w2_s = w2 * s_map; w2_d = w2 * d_map
            w3_s = w3 * s_map; w3_d = w3 * d_map
           
            for y in range(gh):
                for x in range(gw):
                    m0 = ((iy == y) * (ix == x)).float()
                    m1 = ((iy == y) * (ix + 1 == x)).float()
                    m2 = ((iy + 1 == y) * (ix == x)).float()
                    m3 = ((iy + 1 == y) * (ix + 1 == x)).float()
                    
                    static_weights[:, y, x] += (w0_s * m0).sum(-1).sum(-1)
                    static_weights[:, y, x] += (w1_s * m1).sum(-1).sum(-1)
                    static_weights[:, y, x] += (w2_s * m2).sum(-1).sum(-1)
                    static_weights[:, y, x] += (w3_s * m3).sum(-1).sum(-1)

                    dynamic_weights[:, y, x] += (w0_d * m0).sum(-1).sum(-1)
                    dynamic_weights[:, y, x] += (w1_d * m1).sum(-1).sum(-1)
                    dynamic_weights[:, y, x] += (w2_d * m2).sum(-1).sum(-1)
                    dynamic_weights[:, y, x] += (w3_d * m3).sum(-1).sum(-1)

            weights = dynamic_weights / (dynamic_weights + static_weights + 1e-6)
        return weights.unsqueeze(1)
