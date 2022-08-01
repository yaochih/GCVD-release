import torch
import torch.nn as nn
import torch.nn.functional as F

from project_3d import project_pixel
from utils.utils import *


def mean_on_mask(diff, valid_mask, weight=1):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum(-1).sum(-1).sum(-1) / (mask.sum(-1).sum(-1).sum(-1) + 1e-12)
    mean_value = (mean_value * weight).sum()
    return mean_value

class Loss():
    def __init__(self, opt):
        self.opt = opt
        global device
        device = torch.device(opt.cuda)
        self.ssim = SSIM(window_size=3).to(device)

    def photometric_loss(self, target_image, source_image, warp_map):
        rec_image = F.grid_sample(source_image, warp_map)

        diff = (1 - self.opt.ssim_weight) * (target_image - rec_image).abs() + \
               self.opt.ssim_weight * self.ssim(target_image, rec_image)

        return diff.mean(1, keepdim=True), rec_image 

    def optical_flow_loss(self, target_flow, source_flow):
        diff = (target_flow - source_flow).abs().sum(-1).unsqueeze(1)       # L1 distance
        return diff

    def depth_loss(self, depth_projected, depth_estimated, warp_map):
        depth_warped = F.grid_sample(depth_estimated, warp_map)

        if self.opt.loss_depth_mode == 'ratio': # SC-SfMLearner NIPS 2019
            diff = ((depth_projected - depth_warped).abs() /
                    (depth_projected + depth_warped).abs()).clamp(0, 1)
        elif self.opt.loss_depth_mode == 'minmax':
            depth_min = torch.min(depth_projected, depth_warped)
            depth_max = torch.max(depth_projected, depth_warped)
            diff = torch.log(depth_min / depth_max)
        else:
            raise NotImplementedError

        return diff 

    def mesh_loss(self, mesh, weight, base_weight=0.1, adaptive_weight=10.0):
        deform_loss = 0

        if weight.shape[-1] == 1 or weight.shape[-2] == 1: return deform_loss

        loss_v = torch.pow(mesh[..., 1:, :] - mesh[..., :-1, :], 2) * \
                (torch.max(weight[..., 1:, :], weight[..., :-1, :]) * adaptive_weight + base_weight)
        loss_h = torch.pow(mesh[...,    1:] - mesh[...,    :-1], 2) * \
                (torch.max(weight[..., 1:], weight[..., :-1]) * adaptive_weight + base_weight)
        deform_loss = loss_v.sum() + loss_h.sum()
        return deform_loss

    def smoothness_loss(self, image, depth):
        mean_depth = depth.mean(2, keepdim=True).mean(3, keepdim=True)
        norm_depth = (depth / (mean_depth + 1e-12))
        
        depth_grad_x = torch.abs(norm_depth[...,    :-1] - norm_depth[...,    1:])
        depth_grad_y = torch.abs(norm_depth[..., :-1, :] - norm_depth[..., 1:, :])

        image_grad_x = torch.abs(image[...,    :-1] - image[...,    1:]).mean(1, keepdim=True)
        image_grad_y = torch.abs(image[..., :-1, :] - image[..., 1:, :]).mean(1, keepdim=True)

        depth_grad_x *= torch.exp(-image_grad_x)
        depth_grad_y *= torch.exp(-image_grad_y)
        
        depth_grad_x = torch.cat([depth_grad_x, depth_grad_x[...,    -1:].detach() * 0], -1)
        depth_grad_y = torch.cat([depth_grad_y, depth_grad_y[..., -1:, :].detach() * 0], -2)
        
        return depth_grad_x + depth_grad_y


    def depth_grad_loss(self, depth, depth_gt):
        
        depth_grad_x = (depth[..., 1:] - depth[..., :-1])[..., :-1, :]
        depth_grad_y = (depth[..., 1:, :] - depth[..., :-1, :])[..., :-1]

        gt_grad_x = (depth_gt[..., 1:] - depth_gt[..., :-1])[..., :-1, :]
        gt_grad_y = (depth_gt[..., 1:, :] - depth_gt[..., :-1, :])[..., :-1]

        depth_grad = torch.cat((depth_grad_x, depth_grad_y), 1)
        gt_grad = torch.cat((gt_grad_x, gt_grad_y), 1)

        cos_sim = F.cosine_similarity(depth_grad, gt_grad, dim=1)

        grad_loss = torch.pow(1 - cos_sim, 2)
        return grad_loss.mean()

    def compute_pairwise_loss(self, item_a, item_b):

        pose_ab = item_b['pose_inv'] @ item_a['pose']
        
        warp_ab, depth_ab, mask_ab = project_pixel(item_a['depth'], pose_ab, item_b['K'], item_a['K_inv']) 
        dyn_mask_ba = F.grid_sample(item_b['dyn_mask'], warp_ab, padding_mode='border')
        
        # photo
        loss_map_photo_ab, rec_a = self.photometric_loss(item_a['image'], item_b['image'], warp_ab)
        photo_mask_ab = mask_ab * item_a['dyn_mask'] * dyn_mask_ba
        
        # flow
        if 'flow' in item_a.keys():
            flow_ab = item_a['flow']
            loss_map_flow_ab = self.optical_flow_loss(flow_ab, warp_ab)
            flow_mask_ab  = mask_ab * item_a['flow_mask'] * item_a['dyn_mask'] * dyn_mask_ba
        
        # depth
        loss_map_depth_ab = self.depth_loss(depth_ab, item_b['depth'], warp_ab)
        depth_mask_ab = mask_ab * item_a['dyn_mask'] * dyn_mask_ba
        
        # visualization
        item_a['rec'] = rec_a
        if 'flow' in item_a:
            item_a['rec_flow'] = F.grid_sample(item_b['image'], flow_ab)
            item_a['flow_mask']  = flow_mask_ab

        item_a['photo_mask'] = photo_mask_ab
        item_a['depth_mask'] = depth_mask_ab

        loss = {}
        loss['photo'] = mean_on_mask(loss_map_photo_ab, photo_mask_ab, weight=item_a['weight'])
        if 'flow' in item_a.keys():
            loss['flow']  = mean_on_mask(loss_map_flow_ab,  flow_mask_ab,  weight=item_a['weight'])
        loss['depth'] = mean_on_mask(loss_map_depth_ab, depth_mask_ab, weight=item_a['weight'])
        
        return loss
    
    def scale_items(self, items, scale):
        if scale == 1:
            return items
        
        scaled_items = {}

        scaled_items['pose'] = items['pose']
        scaled_items['pose_inv'] = items['pose_inv']

        scaled_items['K'] = items['K'] * 1
        scaled_items['K_inv'] = items['K_inv'] * 1
        scaled_items['K'][:, :2] *= scale
        scaled_items['K_inv'][:, :2, :2] /= scale
        
        for k in items.keys():
            if 'mask' in k or (type(k) is tuple and 'mask' in k[0]):
                scaled_items[k] = F.interpolate(items[k], scale_factor=scale, mode='area')
            elif type(k) is tuple and 'flow' in k[0]:
                scaled_items[k] = F.interpolate(items[k].permute(0,3,1,2), scale_factor=scale, mode='area').permute(0,2,3,1)
            elif 'depth' in k or 'image' in k:
                scaled_items[k] = F.interpolate(items[k], scale_factor=scale, mode='area')

        return scaled_items

    def get_pair_item(self, items, pair):
        
        mode = pair['mode']

        non_flow_keys = ['K', 'K_inv', 'pose', 'pose_inv', 'image', 'depth', 'dyn_mask']
        if 'gt_depth' in items.keys():
            non_flow_keys += ['gt_depth']

        interval = pair['interval']
        if 'a' in pair.keys():
            indices_a, indices_b = pair['a'], pair['b']
            item_a = {k: items[k][indices_a] for k in non_flow_keys}
            item_b = {k: items[k][indices_b] for k in non_flow_keys}
        else:
            item_a = {k: items[k][:-interval] for k in non_flow_keys}
            item_b = {k: items[k][interval:] for k in non_flow_keys}

        if mode == 'seq_flow':
            item_a['flow'] = items[('flow_f', interval)] 
            item_b['flow'] = items[('flow_b', interval)]
            item_a['flow_mask'] = items[('flow_f_mask', interval)]
            item_b['flow_mask'] = items[('flow_b_mask', interval)]
            
            L = item_a['image'].shape[0]
            item_a['weight'] = 1 / L
            item_b['weight'] = 1 / L

        elif mode == 'no_flow':
            L = item_a['image'].shape[0]
            item_a['weight'] = 1 / L
            item_b['weight'] = 1 / L
            for k in non_flow_keys:
                item_b[k] = item_b[k].expand_as(item_a[k])

        return item_a, item_b

    def __call__(self, items, indices_pairs, scales):
        loss = {'full': 0, 'photo': 0, 'flow': 0, 'depth': 0, 'depth_grad': 0} 
        
        vis_a, vis_b = None, None
        
        for s in scales:
            scaled_items = self.scale_items(items, 2 ** -s) 
            
            for pair_i, pair in enumerate(indices_pairs):
                item_a, item_b = self.get_pair_item(scaled_items, pair)

                loss_ab = self.compute_pairwise_loss(item_a, item_b)
                loss_ba = self.compute_pairwise_loss(item_b, item_a)

                for k in loss_ab.keys():
                    loss[k] += (loss_ab[k] + loss_ba[k]) * 0.5 * pair['weight']

                if pair['mode'] == 'seq_flow':
                    vis_a, vis_b = item_a, item_b


        loss['full'] += loss['photo'] * self.opt.loss_photo + \
                        loss['flow'] *  self.opt.loss_flow + \
                        loss['depth'] * self.opt.loss_depth

        if self.opt.loss_depth_grad > 0:
            # target depth supervision
            w_grad = self.opt.loss_depth_grad * 5 if indices_pairs[0]['grad'] else self.opt.loss_depth_grad
            loss['depth_grad'] = 0
            for s in [0, 1, 2]:
                depth = F.interpolate(items['depth'], scale_factor=2**-s, mode='bilinear')
                gt_depth = F.interpolate(items['gt_depth'], scale_factor=2**-s, mode='bilinear')
                loss['depth_grad'] += self.depth_grad_loss(depth, gt_depth) * (2 ** s) / 7
            loss['full'] += loss['depth_grad'] * w_grad
        
        if self.opt.loss_smooth > 0:
            # handle dynamic regions
            loss['smooth'] = mean_on_mask(self.smoothness_loss(items['image'], items['depth']), items['dyn_mask'])
            loss['full'] += loss['smooth'] * self.opt.loss_smooth

        if self.opt.mesh_deformation:
            # mesh deformation
            loss['mesh'] = self.mesh_loss(items['mesh'], items['mesh_weight'])
            loss['full'] += loss['mesh'] * self.opt.loss_mesh

        return loss, vis_a, vis_b

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, window_size=3, alpha=1, beta=1, gamma=1):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(window_size, 1)
        self.mu_y_pool   = nn.AvgPool2d(window_size, 1)
        self.sig_x_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_y_pool  = nn.AvgPool2d(window_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(window_size, 1)

        self.refl = nn.ReflectionPad2d(window_size//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.C3 = self.C2 / 2
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if alpha == 1 and beta == 1 and gamma == 1:
            self.run_compute = self.compute_simplified
        else:
            self.run_compute = self.compute
        

    def compute(self, x, y):
        
        x = self.refl(x)
        y = self.refl(y)
        
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        l = (2 * mu_x * mu_y + self.C1) / \
            (mu_x * mu_x + mu_y * mu_y + self.C1)
        c = (2 * sigma_x * sigma_y + self.C2) / \
            (sigma_x + sigma_y + self.C2)
        s = (sigma_xy + self.C3) / \
            (torch.sqrt(sigma_x * sigma_y) + self.C3)

        ssim_xy = torch.pow(l, self.alpha) * \
                  torch.pow(c, self.beta) * \
                  torch.pow(s, self.gamma)
        return torch.clamp((1 - ssim_xy) / 2, 0, 1)

    def compute_simplified(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

    def forward(self, x, y):
        return self.run_compute(x, y)


