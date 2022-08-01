import torch

import numpy as np
import cv2 as cv
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def normalize_for_grid_sample(flow):
    warp_map = flow.clone().detach()
    H, W = warp_map.size()[-2:]
    warp_map[:, 0, :, :] = warp_map[:, 0, :, :] / (W - 1) * 2 - 1
    warp_map[:, 1, :, :] = warp_map[:, 1, :, :] / (H - 1) * 2 - 1
    return warp_map.permute(0, 2, 3, 1)

def get_grid(tensor, homogeneous=False):
    B, _, H, W = tensor.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(tensor.device)
    if homogeneous:
        ones = torch.ones(B, 1, H, W).float().to(tensor.device)
        grid = torch.cat((grid, ones), 1)
    return grid

def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}
def depths2show(depths, max_value=None, colormap='magma', inverse=True):
    depths = depths.squeeze(1) 
    B, H, W = depths.shape
    depths = depths.detach().cpu().numpy()
    if inverse:
        depths = 1. / depths

    min_depth = depths.min()#(axis=-1, keepdims=True).min(axis=-2, keepdims=True)
    max_depth = depths.max()#(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
    depths_norm = (depths - min_depth) / (max_depth - min_depth)
    depths_norm = depths_norm.reshape(-1, W)
    depths_ = COLORMAPS[colormap](depths_norm).astype(np.float32)[:,:,:-1]
    depths_ = torch.from_numpy(depths_).view(B, H, W, 3).permute(0, 3, 1, 2)
    
    return depths_

