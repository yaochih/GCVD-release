import sys, os, argparse, glob
import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F
from PIL import Image

from RAFT.core.raft import RAFT
from utils.utils import get_grid, normalize_for_grid_sample
from logger import print_text, progress_bar

RAFT_PRETRAINED = 'weights/models/raft-things.pth'

class FlowProcessor(torch.nn.Module):
    def __init__(self, opt):
        super(FlowProcessor, self).__init__()
        
        self.device = torch.device(opt.cuda)
        model = torch.nn.DataParallel(RAFT()).to(self.device)
        model.load_state_dict(torch.load(RAFT_PRETRAINED))
        
        self.model = model.module
        self.model = self.model.to(self.device)
        self.model.eval()

    def load_image(self, imfile):
        image = np.array(Image.open(imfile)).astype(np.uint8)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image[None].to(self.device)

    def estimate_homography(self, im1, im2):
        B = im1.size(0)
        batch_H = []
        inlier_ratios = []
        for b in range(B):
            im1_ = ((im1[b].cpu().numpy().transpose(1, 2, 0) * 255.)).astype(np.uint8)
            im2_ = ((im2[b].cpu().numpy().transpose(1, 2, 0) * 255.)).astype(np.uint8)

            im1_ = im1_[..., -1::]
            im2_ = im2_[..., -1::]

            sift = cv.SIFT_create()
            kp1, des1 = sift.detectAndCompute(im1_, None)
            kp2, des2 = sift.detectAndCompute(im2_, None)

            try:
                matcher = cv.BFMatcher()
                matches = matcher.knnMatch(des1, des2, k=2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                if len(good) < 20:
                    H = np.eye(3)
                else:
                    H, mask_H = cv.findHomography(pts1, pts2, method=cv.RANSAC)
                    F, mask_F = cv.findFundamentalMat(pts1, pts2, method=cv.RANSAC)
                    in_ratio = max(mask_H.mean(), mask_F.mean())
                    inlier_ratios.append(in_ratio)
                    if mask_H.mean() < 0.4: 
                        H = np.eye(3)
            except:
                H = np.eye(3)
                inlier_ratios.append(0)

            batch_H.append(H)

        batch_H = torch.FloatTensor(np.stack(batch_H, 0)).to(im1.device)
        return batch_H, inlier_ratios

    def pre_warping(self, im, homo):
        grid = get_grid(im, homogeneous=True)
        warp_coord = homo @ grid.view(im.size(0), 3, -1)
        warp_coord = warp_coord.view(im.size(0), 3, im.size(2), im.size(3))
        warp_coord[:, 0] /= warp_coord[:, 2]
        warp_coord[:, 1] /= warp_coord[:, 2]
        warp_coord = warp_coord[:, :2]
        warp_coord_norm = warp_coord.clone()
        warp_coord_norm[:, 0, :, :] = warp_coord[:, 0, :, :] / (im.size(-1) - 1) * 2 - 1
        warp_coord_norm[:, 1, :, :] = warp_coord[:, 1, :, :] / (im.size(-2) - 1) * 2 - 1
        im_warped = F.grid_sample(im, warp_coord_norm.permute(0, 2, 3, 1))
        return im_warped, warp_coord - grid[:, :2]

    def get_flow(self, im1, im2, backward=False, grid_normalize=False, pre_homo=True):
        with torch.no_grad():
            grid = get_grid(im1)
            H, W = im1.size()[-2:]
            if pre_homo is not False:
                if pre_homo is True:
                    homo12, inlier_ratios = self.estimate_homography(im1, im2)
                else:
                    homo12 = pre_homo

                if backward:
                    homo21 = torch.inverse(homo12)

                if not backward:
                    #homo = self.estimate_homography(im1, im2)
                    im2_homo_warped, homo_warp = self.pre_warping(im2, homo12)
                    flow, _ = self.model(im1, im2_homo_warped, iters=20, test_mode=True)
                    homo = homo12
                else:
                    #homo = self.estimate_homography(im2, im1)
                    im1_homo_warped, homo_warp = self.pre_warping(im1, homo21)
                    flow, _ = self.model(im2, im1_homo_warped, iters=20, test_mode=True)
                    homo = homo21

                flow_norm = flow + grid
                flow_norm = normalize_for_grid_sample(flow_norm)
                homo_warp = F.grid_sample(homo_warp, flow_norm)

                flow += homo_warp
            else:
                if not backward:
                    flow, _ = self.model(im1, im2, iters=20, test_mode=True)
                else:
                    flow, _ = self.model(im2, im1, iters=20, test_mode=True)

        flow += grid
        if grid_normalize:
            flow = normalize_for_grid_sample(flow)
        if pre_homo is True:
            return flow, homo12, inlier_ratios
        else:
            return flow

    def get_flow_forward_backward(self, im1, im2, pre_homo=True, consistency_thresh=1.0):
        inlier_ratios = []
        if pre_homo:
            flow12, homo12, inlier_ratios = self.get_flow(im1, im2, pre_homo=pre_homo)
            flow21 = self.get_flow(im1, im2, pre_homo=homo12, backward=True)
        else:
            flow12 = self.get_flow(im1, im2, pre_homo=False)
            flow21 = self.get_flow(im1, im2, pre_homo=False, backward=True)
        mask1 = self.get_consistency_map(flow12, flow21, consistency_thresh)
        mask2 = self.get_consistency_map(flow21, flow12, consistency_thresh)

        return flow12, flow21, mask1, mask2, inlier_ratios

    def get_consistency_map(self, flow12, flow21, consistency_thresh=1.0):
        flow21_warped = F.grid_sample(flow21, normalize_for_grid_sample(flow12))

        diff = flow21_warped - get_grid(flow12)
        diff = torch.sqrt(torch.sum(torch.pow(diff, 2), 1))
        mask1 = diff < consistency_thresh
        mask1 = mask1.float()

        return mask1

    def compute_sequence(self, input_frames, output_dir, pre_homo=True, consistency_thresh=1.0, intervals=[1]):
        h, w = self.load_image(input_frames[0]).shape[-2:]
        
        for i in intervals:
            output_dir_i = output_dir/str(i)
            output_dir_i.makedirs_p()
            pbar = progress_bar(len(input_frames) - i)

            for j in range(len(input_frames) - i):
                img1 = self.load_image(input_frames[j])
                img2 = self.load_image(input_frames[j + i])

                flow12, flow21, mask1, mask2, _ = self.get_flow_forward_backward(
                        img1, img2, pre_homo=pre_homo, consistency_thresh=consistency_thresh)

                package = torch.cat([flow12, flow21, mask1.unsqueeze(0), mask2.unsqueeze(0)], 1)
                np.save(output_dir_i/os.path.split(input_frames[j])[-1], package.detach().cpu().numpy())
                pbar.update(1)
            pbar.close()

    def compute_chain_flow(self, flows_f, flows_b):
        length = len(flows_f)
        flow_f = flows_f[-2]
        for i in range(length - 3, -1, -1):
            flow_f = F.grid_sample(flow_f, normalize_for_grid_sample(flows_f[i]))
        
        flow_b = flows_b[1]
        for i in range(2, length):
            flow_b = F.grid_sample(flow_b, normalize_for_grid_sample(flows_b[i]))

        mask1, mask2 = self.get_consistency_map(flow_f, flow_b)
        return flow_f, flow_b, mask1, mask2

    def rescale_flow(self, flow, target_size):
        h_, w_ = target_size
        h, w = flow.shape[-2:]
        flow[:, 0] = flow[:, 0] / w * w_
        flow[:, 1] = flow[:, 1] / h * h_
        flow = F.interpolate(flow, target_size)
        return flow
