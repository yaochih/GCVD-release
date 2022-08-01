# Most from https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/inverse_warp.py

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
import sys

from utils.utils import *

def project_pixel(depth, pose, K, K_inv, depth_min=1e-5):
    B, _, H, W = depth.shape
    cam_coords = K_inv @ get_grid(depth, homogeneous=True).reshape(B, 3, -1)
    cam_coords = cam_coords * depth.reshape(B, 1, -1)
    
    proj_cam_to_src_pixel = K @ pose[:, :3]
    R = proj_cam_to_src_pixel[:, :, :3]
    t = proj_cam_to_src_pixel[:, :, -1:]

    pcoords = R @ cam_coords + t

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=depth_min)
    
    X_norm = 2 * (X / Z) / (W - 1) - 1
    Y_norm = 2 * (Y / Z) / (H - 1) - 1

    warp_map = torch.stack([X_norm, Y_norm], dim=2) # [B, H*W, 2]
    
    valid_points = warp_map.abs().max(dim=-1)[0] <= 1
    mask = valid_points.unsqueeze(1).float()

    return warp_map.reshape(B, H, W, 2), Z.reshape(B, 1, H, W), mask.reshape(B, 1, H, W)


def intrinsic_vec2mat(f, cx, cy):
    B = f.size(0)
    zeros = f.detach() * 0
    ones = zeros.detach() + 1
    cx = ones.detach() * cx
    cy = ones.detach() * cy
    f_inv = f.reciprocal()
    K = torch.stack([f, zeros, cx,
                     zeros, f, cy,
                     zeros, zeros, ones], 1).reshape(B, 3, 3)
    K_inv = torch.stack([f_inv, zeros, -cx * f_inv,
                         zeros, f_inv, -cy * f_inv,
                         zeros, zeros, ones], 1).reshape(B, 3, 3)
    return K, K_inv

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    #norm_quat = torch.nn.functional.normalize(quat, p=2, dim=1)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:6]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    
    bot = (transform_mat[:, -1, :].detach() * 0.).view(-1, 1, 4)
    bot[:, :, -1] += 1.
    transform_mat = torch.cat([transform_mat, bot], dim=1)

    return transform_mat

def inverse_pose(pose_mat):
    
    R = pose_mat[:, :3, :3]
    t = pose_mat[:, :3, 3:]

    R_T = torch.transpose(R, 1, 2)
    t_inv = -R_T @ t
    pose_inv = torch.cat([R_T, t_inv], dim=2)
    
    bot = (pose_inv[:, -1, :].detach() * 0.).view(-1, 1, 4)
    bot[:, :, -1] += 1.
    pose_inv = torch.cat([pose_inv, bot], dim=1)
    return pose_inv

def inverse_intrinsic(K):
    K_inv = K.clone()
    fx_inv = K[:, 0, 0].reciprocal()
    fy_inv = K[:, 1, 1].reciprocal()
    K_inv[:, 0, 0] = fx_inv
    K_inv[:, 1, 1] = fy_inv
    K_inv[:, 0, 2] = -K[:, 0, 2] * fx_inv
    K_inv[:, 1, 2] = -K[:, 1, 2] * fy_inv
    return K_inv

def rotation_weight_sum(R1, R2, weight1):
    r = Rotation.from_matrix([R1, R2])
    return r.mean(weights=[weight1, 1-weight1]).as_euler('xyz')
