import sys, os, time
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tvutils
import optimizer
from loss import Loss
from utils.utils import *
from project_3d import *
from logger import print_text, progress_bar

class SFM:
    def __init__(self, models, seq_io, opt):
        self.opt = opt
        global device
        device = torch.device(opt.cuda)

        self.models = models
        self.seq_io = seq_io
        self.loss = Loss(opt)
        
        if opt.log_tensorboard:
            log_dir = seq_io.root/'log'
            log_dir.makedirs_p()
            self.log_writer = SummaryWriter(log_dir)

        self.weight_dir = seq_io.root/'weights'
        self.weight_dir.makedirs_p()

    def create_optimizer(self, learning_rate):
        self.models['depth'].train()
        self.models['pose'].train()
        
        depth_params = filter(lambda p: p.requires_grad, self.models['depth'].parameters()) 
        pose_params =  filter(lambda p: p.requires_grad, self.models['pose'].parameters())
        
        train_params = [
            {'params': depth_params, 'lr': learning_rate},
            {'params': pose_params,  'lr': learning_rate}
        ]
        
        if self.opt.mesh_deformation:
            self.models['mesh'].train()
            train_params.append({'params': self.models['mesh'].parameters(), 'lr': learning_rate * 100})

        self.optimizer = optimizer.create(
            self.opt.optimizer,
            train_params,
            learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0
        )

    def reconstruct_images(self, 
            items, 
            video_indices, 
            depth_indices, 
            pose_indices, 
            loss_pairs, 
            num_epochs,
            lr,
            scales,
            pose_base=True,
            log_name='test', 
            pbar=None):
        
        if self.opt.mesh_deformation:
            items['mesh_weight'] = self.models['mesh'].get_weight_map(items['dyn_mask'][depth_indices])

        self.create_optimizer(lr)

        if len(depth_indices) > 0:
            self.models['depth'].set_feature(items['image'][depth_indices])

        if len(pose_indices) > 0:
            if pose_base:
                base = min(pose_indices) - 1
                if base >= 0: pose_indices = [base] + pose_indices
                else: base = 0
    
            self.models['pose'].set_feature(items['image'][pose_indices], indices=video_indices[pose_indices])
       
        for ep in range(1, num_epochs + 1):
            out = {}
            
            for k in ['depth', 'pose', 'pose_inv', 'K', 'K_inv']:
                items[k] = items[k].detach()
            
            if len(depth_indices) > 0:
                out['depth'] = self.models['depth'](items['image'][depth_indices])
                items['depth'][depth_indices] = out['depth']
             
            if self.opt.mesh_deformation:
                items['mesh'] = self.models['mesh'].mesh
                items['depth'][depth_indices] = self.models['mesh'](items['depth'][depth_indices])

            if len(pose_indices) > 0:
                out['pose'], out['pose_inv'], out['K'], out['K_inv']= self.models['pose'](
                    items['image'][pose_indices], indices=video_indices[pose_indices])

                if pose_base:
                    out['pose'] = out['pose_inv'][:1].expand_as(out['pose']) @ out['pose']
                    items['pose'][pose_indices] = items['pose'][base:base+1].expand_as(out['pose']) @ out['pose']
                else:
                    items['pose'][pose_indices] = out['pose']
                items['pose_inv'] = torch.inverse(items['pose'])
                items['K'][pose_indices] = out['K']
                items['K_inv'][pose_indices]   = out['K_inv']
            
            self.optimizer.zero_grad()
            loss, items_vis1, item_vis2 = self.loss(items, loss_pairs, scales)
            loss['full'].backward()
            self.optimizer.step()

            if self.opt.log_tensorboard:
                if ep == 1 or ep == num_epochs or (ep % self.opt.log_freq) == 0:
                    self.log_results(items_vis1, loss, ep, log_name)
            
            if pbar is not None:
                pbar.update(1)
        
        return items

    def save_weights(self, name):
        save_dir = self.weight_dir/name
        save_dir.makedirs_p()
        self.models['depth'].save_weight(save_dir/'depth.pth')
        self.models['pose'].save_weight(save_dir/'pose.pth')

    def load_weights(self, name):
        self.models['depth'].load_weight(self.weight_dir/name/'depth.pth')
        self.models['pose'].load_weight(self.weight_dir/name/'pose.pth')

    def merge_weights(self, name):
        self.models['depth'].merge_weight(self.weight_dir/name/'depth.pth')
        self.models['pose'].merge_weight(self.weight_dir/name/'pose.pth')

    def log_results(self, items, loss, ep, name):
        if items is None: return

        if self.opt.log_image_scale > 0:
            num_images_to_show = items['image'].shape[0] 
            
            images_to_show = items['image'][:num_images_to_show].clone().detach().cpu()
            rec_f_to_show = items['rec'][:num_images_to_show].clone().detach().cpu()
            flow_f_to_show = items['rec_flow'][:num_images_to_show].clone().detach().cpu()
            flow_mask_to_show = items['flow_mask'][:num_images_to_show].clone().detach().cpu().expand(-1, 3, -1, -1)
            p_mask_to_show = items['photo_mask'][:num_images_to_show].clone().detach().cpu().expand(-1, 3, -1, -1)
            f_mask_to_show = items['flow_mask'][:num_images_to_show].clone().detach().cpu().expand(-1, 3, -1, -1)
            d_mask_to_show = items['depth_mask'][:num_images_to_show].clone().detach().cpu().expand(-1, 3, -1, -1)
            depths_to_show = depths2show(items['depth'][:num_images_to_show])
            
            grid_image = tvutils.make_grid(F.interpolate(torch.cat((
                images_to_show, 
                rec_f_to_show,
                flow_f_to_show,
                flow_mask_to_show,
                depths_to_show,
                ), 0), scale_factor=self.opt.log_image_scale), nrow=num_images_to_show)
        
        try: self.log_writer.add_image('{}/depth'.format(name), grid_image, ep)
        except: pass
        try: self.log_writer.add_scalar('{}/loss/total'.format(name), loss['full'], ep)
        except: pass
        try: self.log_writer.add_scalar('{}/loss/photo'.format(name), loss['photo'].item(), ep)
        except: pass
        try: self.log_writer.add_scalar('{}/loss/flow'.format(name),  loss['flow'].item(), ep)
        except: pass
        try: self.log_writer.add_scalar('{}/loss/depth'.format(name), loss['depth'].item(), ep)
        except: pass
        if 'smooth' in loss.keys():
            try: self.log_writer.add_scalar('{}/loss/smooth'.format(name), loss['smooth'].item(), ep)
            except: pass
        if 'mesh' in loss.keys():
            try: self.log_writer.add_scalar('{}/loss/mesh'.format(name), loss['mesh'].item(), ep)
            except: pass
        if 'depth_grad' in loss.keys():
            try: self.log_writer.add_scalar('{}/loss/depth_grad'.format(name), loss['depth_grad'].item(), ep)
            except: pass
