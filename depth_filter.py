import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from utils.utils import *
from logger import print_text, progress_bar

class DepthFilter:
    def __init__(self, opt, seq_io):
        self.opt = opt
        self.seq_io = seq_io

    def get_3D_points(self, depth, K_inv):
        B, _, H, W = depth.shape
        cam_coords = K_inv @ get_grid(depth, homogeneous=True).reshape(B, 3, -1)
        cam_coords = cam_coords * depth.reshape(B, 1, -1)
        cam_coords = cam_coords.reshape(B, 3, H, W)
        return cam_coords
        
    def warp_project(self, points, pose, flow, mask=None):
        B, H, W = flow.shape[:-1]
        R = pose[:, :3, :3]
        t = pose[:, :3, -1:]
        pcoords = R @ points.reshape(B, 3, -1) + t

        Z = pcoords[:, 2].reshape(B, 1, H, W)
        #Z *= mask
        warped_Z = F.grid_sample(Z, flow, mode='bilinear')
        return warped_Z

    def compute_chaining_flow(self, segments, intervals):
        max_span = max(intervals)
        for i in range(2, max_span + 1):
            flow_f = segments[('flow_f', i - 1)]
            flow_f_valid = (flow_f[:-1].abs().max(dim=-1)[0] <= 1)
            segments[('flow_f', i)] = F.grid_sample(flow_f[1:].permute(0,3,1,2), flow_f[:-1]).permute(0,2,3,1)
            segments[('flow_f', i)][flow_f_valid == False, :] = 2

            flow_b = segments[('flow_b', i - 1)]
            flow_b_valid = (flow_b[1:].abs().max(dim=-1)[0] <= 1)
            segments[('flow_b', i)] = F.grid_sample(flow_b[:-1].permute(0,3,1,2), flow_b[1:]).permute(0,2,3,1)
            segments[('flow_b', i)][flow_b_valid == False, :] = 2

        return segments

    def get_flow_diff(self, segments, intervals):
        max_span = max(intervals)
        for i in range(1, max_span + 1):
            flow_f = segments[('flow_f', i)]
            flow_b = segments[('flow_b', i)]

            flow12 = flow_f.clone()
            flow21 = flow_b.clone()
            _, H, W, _ = flow12.shape
            flow12[..., 0] = (flow12[..., 0] + 1) * 0.5 * (W - 1)
            flow12[..., 1] = (flow12[..., 1] + 1) * 0.5 * (H - 1)
            flow21[..., 0] = (flow21[..., 0] + 1) * 0.5 * (W - 1)
            flow21[..., 1] = (flow21[..., 1] + 1) * 0.5 * (H - 1)
            flow12 = flow12.permute(0, 3, 1, 2)
            flow21 = flow21.permute(0, 3, 1, 2)
            flow21_warped = F.grid_sample(flow21, normalize_for_grid_sample(flow12), mode='bilinear')
            flow12_warped = F.grid_sample(flow21, normalize_for_grid_sample(flow21), mode='bilinear')
            
            diff_f = flow21_warped - get_grid(flow12) + 1e-12
            diff_f = torch.sqrt(torch.sum(torch.pow(diff_f, 2), 1, keepdim=True))
            
            diff_b = flow12_warped - get_grid(flow21) + 1e-12
            diff_b = torch.sqrt(torch.sum(torch.pow(diff_b, 2), 1, keepdim=True))
            segments[('flow_f_diff', i)] = diff_f
            segments[('flow_b_diff', i)] = diff_b
        return segments

    def process_sequence(self):
        i = 0 
        max_span = 4
        pbar = progress_bar(self.seq_io.length)
        with torch.no_grad():
            while i < self.seq_io.length:
                end = min(i + self.opt.segment_max_batch_size * 2, self.seq_io.length)
                self.process_segment(i, end, max_span)
                if end == self.seq_io.length: 
                    pbar.update(end - i)
                    break

                new_i = end - max_span * 2
                pbar.update(new_i - i)
                i = new_i
    
    def process_segment(self, begin_index, end_index, max_span):
        indices = list(range(begin_index, end_index))
        segments = self.seq_io.get_items(indices, load_depth=True, load_camera=True, load_subdir=self.opt.save_subdir)

        target_depth = segments['depth']
        crop = 4 
        target_depth = target_depth[:, :, crop:-crop, crop:-crop]
        target_depth = F.pad(target_depth, (crop, crop, crop, crop), 'replicate')
        pts_3d = self.get_3D_points(target_depth, segments['K_inv'])

        intervals = list(range(-max_span, max_span + 1))
        weights = torch.zeros(len(indices), len(intervals), pts_3d.shape[-2], pts_3d.shape[-1]).to(pts_3d.device)# + 1e-6
        depths = torch.zeros(len(indices), len(intervals), pts_3d.shape[-2], pts_3d.shape[-1]).to(pts_3d.device)# + 1e-9
        
        g_weights = gaussian(len(intervals), len(intervals)//4)

        segments = self.compute_chaining_flow(segments, intervals)
        segments = self.get_flow_diff(segments, intervals)
        beta1, beta2 = -2, -0.1
        max_span = max(intervals)
        for i in range(1, max_span + 1):
            i_p = intervals.index(i)
            i_m = intervals.index(-i)

            warped_depth_f = self.warp_project(pts_3d[i:], 
                    segments['pose_inv'][:-i] @ segments['pose'][i:], 
                    segments[('flow_f', i)])
            mask_f = (warped_depth_f > 0)
            weight_f = torch.max(target_depth[:-i], warped_depth_f) / torch.min(target_depth[:-i], warped_depth_f)
            weight_f = torch.exp(beta1 * weight_f + beta2 * (segments[('flow_f_diff', i)] + 1))
            weights[:-i, i_p:i_p+1] = weight_f
            depths[:-i, i_p:i_p+1] = warped_depth_f

            warped_depth_b = self.warp_project(pts_3d[:-i], 
                    segments['pose_inv'][i:] @ segments['pose'][:-i], 
                    segments[('flow_b', i)])
            mask_b = (warped_depth_b > 0)
            weight_b = torch.max(target_depth[i:], warped_depth_b) / torch.min(target_depth[i:], warped_depth_b) 
            weight_b = torch.exp(beta1 * weight_b + beta2 * (segments[('flow_b_diff', i)] + 1))
            weights[i:, i_m:i_m+1] = weight_b
            depths[i:, i_m:i_m+1] = warped_depth_b
        
        i_t = intervals.index(0)
        weights[:, i_t] = np.exp(beta1 + beta2)
        depths[:, i_t:i_t+1] = target_depth
        
        sum_depth = (depths * weights).sum(1, keepdim=True)
        sum_weight = weights.sum(1, keepdim=True)

        filtered_depth = sum_depth / sum_weight
        if begin_index > 0:
            indices = indices[max_span:]
            filtered_depth = filtered_depth[max_span:]
        if end_index < self.seq_io.length:
            indices = indices[:-max_span]
            filtered_depth = filtered_depth[:-max_span]
        self.seq_io.save_items(indices, {'depth': filtered_depth}, save_subdir='filtered')

        
