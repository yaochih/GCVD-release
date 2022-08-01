import random, os, sys, glob, subprocess
import torch
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
from imageio import imread, imwrite
from path import Path
import cv2 as cv
from skimage.transform import resize as imresize

from utils.utils import *
from flow import FlowProcessor
from project_3d import *
from logger import print_text, progress_bar

midas_weight_path = 'weights/models/midas_v21-f6b98070.pt'

class SequenceIO(data.Dataset):
    def __init__(self, opt, preprocess=True):
        self.opt = opt
        global device
        device = torch.device(opt.cuda)
        self.input_video = opt.input_path
        self.root = Path(opt.output_dir)/opt.name
        
        self.root.makedirs_p()
        self.image_dir = {}
        self.image_dir['full'] = self.root/'images'
        self.image_dir['flow'] = self.root/'images_flow'
        self.image_dir['down'] = self.root/'images_down'
        self.image_dir['key'] = self.root/'images_keyframe'
        self.flow_dir = {}
        self.flow_dir['seq'] = self.root/'flow'/'sequence'
        self.flow_dir['key'] = self.root/'flow'/'keyframe'
        self.flow_dir['seg'] = self.root/'flow'/'segment'
        self.flow_dir['graph'] = self.root/'flow'/'graph'
        self.dynamic_mask_dir = self.root/'dynamic_mask'
        self.target_depth_dir = self.root/'midas_depth'
        self.target_depth_ext = '*.pfm'
        self.depth_dir =  self.root/'depths'
        self.camera_dir = self.root/'camera'
        
        self.preprocess = preprocess
        self.extract_frames()
        self.load_video_info()
        self.downsample_frames()
        self.generate_dynamic_mask()
        self.generate_target_depth()
        self.compute_flow()
        
        self.load_keyframes()

    ################### Key frame ####################
    @torch.no_grad()
    def decide_keyframes(self):
        self.shift_thresh = self.opt.keyframe_thresh * min(self.image_size['flow'])
        shift_sum = 0
        self.keyframes = []

        print_text('Keyframe decision', level='H2', verbose=2)
        pbar = progress_bar(self.length)

        for i in range(0, self.length):
            dyn_mask = self.load_dynamic_mask(i).unsqueeze(0)
            mask = F.interpolate(dyn_mask, self.image_size['flow'], mode='area').squeeze(0)
            
            if i < self.length - 1:
                flow_f, flow_b, flow_mask, _ = self.load_sequence_flow(i)
                flow_diff = flow_f - get_grid(flow_f)
                flow_mag = torch.sqrt(torch.sum(torch.pow(flow_diff, 2), 1))
                mean_mag = (flow_mag * mask).sum() / mask.sum()
                shift_sum += mean_mag 

            if shift_sum >= self.shift_thresh or i == 0 or i == self.length - 1 or \
                    (len(self.keyframes) > 0 and i - self.keyframes[-1] >= self.opt.segment_max_batch_size - 1):
                self.keyframes.append(i)
                shift_sum = 0

                image = imread(self.image_dir['flow']/self.image_names[i])
                mask = mask.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
                image *= mask
                imwrite(self.image_dir['key']/self.image_names[i], image.astype(np.uint8))   
            
            if pbar is not None:
                pbar.update(1)

        np.savetxt(self.root/'keyframes.txt', self.keyframes)
        
    def load_keyframes(self):
        self.shift_thresh = self.opt.keyframe_thresh * min(self.image_size['flow'])
        self.keyframes = np.genfromtxt(self.root/'keyframes.txt').astype(int)
        
        print_text('Number of Keyframes: {}'.format(len(self.keyframes)), 'H2', verbose=2)
        print_text('Keyframes indexs: \n{}'.format(self.keyframes), 'H2', verbose=3, box=True)

    ################### optical flow ####################
    def compute_flow(self):
        
        if not self.preprocess and not self.opt.regenerate_keyframe: return
        flow = FlowProcessor(self.opt).to(device)
        
        # compute flow between adjacent frames
        self.flow_dir['seq'].makedirs_p()
        image_paths = sorted(list(glob.glob(self.image_dir['flow']/'*.{}'.format(self.opt.image_ext))))
        if self.preprocess:
            flow.compute_sequence(image_paths, self.flow_dir['seq'], pre_homo=False, consistency_thresh=1.0)
        
        self.decide_keyframes()
        self.load_keyframes()

        # compute flow of keyframes
        self.flow_dir['key'].makedirs_p()
        image_paths = [self.image_dir['key']/'{}'.format(self.image_names[i]) for i in self.keyframes]
        flow.compute_sequence(image_paths, self.flow_dir['key'], pre_homo=True, consistency_thresh=2.0, 
                intervals=[1])
        
        del flow

    def load_graph_flow(self, tgt, ref, load_size='down'):
        flow_item = np.load(self.flow_dir['graph']/'{}_{}.npy'.format(tgt, ref), allow_pickle=True)[()]
        flow_f = flow_item['flow_f'].to(device)
        flow_b = flow_item['flow_b'].to(device)
        flow_f = normalize_for_grid_sample(flow_f).permute(0, 3, 1, 2)
        flow_b = normalize_for_grid_sample(flow_b).permute(0, 3, 1, 2)
        flow_f = F.interpolate(flow_f, self.image_size[load_size], mode='area').permute(0, 2, 3, 1)
        flow_b = F.interpolate(flow_b, self.image_size[load_size], mode='area').permute(0, 2, 3, 1)
        mask_f = F.interpolate(flow_item['mask_f'].unsqueeze(0), self.image_size[load_size], mode='area')
        mask_b = F.interpolate(flow_item['mask_b'].unsqueeze(0), self.image_size[load_size], mode='area')
        return flow_f, flow_b, mask_f, mask_b

    def load_sequence_flow(self, index, interval=1, folder='seq'):
        flow_item = np.load(self.flow_dir[folder]/str(interval)/self.image_names[index] + '.npy')
        flow_item = torch.from_numpy(flow_item).float().to(device)
        flow_f = flow_item[:, :2]
        flow_b = flow_item[:, 2:4]
        flow_f_mask = flow_item[:, 4:5]
        flow_b_mask = flow_item[:, 5:]
        return flow_f, flow_b, flow_f_mask, flow_b_mask

    def load_sequence_flows(self, indices, interval=1, load_size='down', folder='seq'):
        flows_f, flows_b, flows_fm, flows_bm = [], [], [], []
        for i in indices[:-interval]:
            flow_f, flow_b, flow_fm, flow_bm = self.load_sequence_flow(i, interval, folder=folder)
            flows_f.append(flow_f)
            flows_b.append(flow_b)
            flows_fm.append(flow_fm)
            flows_bm.append(flow_bm)

        flows_f = torch.cat(flows_f, 0)
        flows_b = torch.cat(flows_b, 0)
        flows_fm = torch.cat(flows_fm, 0)
        flows_bm = torch.cat(flows_bm, 0)

        flows_f = normalize_for_grid_sample(flows_f).permute(0, 3, 1, 2)
        flows_b = normalize_for_grid_sample(flows_b).permute(0, 3, 1, 2)
        flows_f = F.interpolate(flows_f, self.image_size[load_size], mode='area').permute(0, 2, 3, 1)
        flows_b = F.interpolate(flows_b, self.image_size[load_size], mode='area').permute(0, 2, 3, 1)
        flows_fm = F.interpolate(flows_fm, self.image_size[load_size], mode='area')
        flows_bm = F.interpolate(flows_bm, self.image_size[load_size], mode='area')
        
        return flows_f, flows_b, flows_fm, flows_bm
    
    def compute_flow_mask(self, flow_f, flow_b, thresh=3):
        mask = torch.ones(flow_f[..., 0].shape).to(flow_f.device)
        H, W = mask.shape[-2:]
        flow_f_ = flow_f.clone()
        flow_b_ = flow_b.clone()

        flow_f_[..., 0] = (flow_f_[..., 0] + 1) * 0.5 * (W - 1)
        flow_f_[..., 1] = (flow_f_[..., 1] + 1) * 0.5 * (H - 1)
        flow_b_[..., 0] = (flow_b_[..., 0] + 1) * 0.5 * (W - 1)
        flow_b_[..., 1] = (flow_b_[..., 1] + 1) * 0.5 * (H - 1)
        flow_f_ = flow_f_.permute(0, 3, 1, 2)
        flow_b_ = flow_b_.permute(0, 3, 1, 2)
        grid = get_grid(flow_f_[:-1])
        flow_diff_f = torch.sqrt(torch.sum(torch.pow(grid - F.grid_sample(flow_b_[1:], flow_f[:-1]), 2), 1))
        flow_diff_b = torch.sqrt(torch.sum(torch.pow(grid - F.grid_sample(flow_f_[:-1], flow_b[1:]), 2), 1))

        mask[:-1] *= (flow_diff_f < thresh)
        mask[1:]  *= (flow_diff_b < thresh)
        return mask.unsqueeze(1)

    ################### Mask of dynamic object ####################
    def generate_dynamic_mask(self):
        import dynamic_mask_generation
        if not self.preprocess: return
        
        self.dynamic_mask_dir.makedirs_p()
        args, _ = dynamic_mask_generation.get_parser().parse_known_args()
        args.input = [self.image_dir['full']/'*.{}'.format(self.opt.image_ext)]
        args.output = self.dynamic_mask_dir

        dynamic_mask_generation.dynamic_mask_generation(args)

    def load_dynamic_mask(self, index):
        return (torch.from_numpy(imread(self.dynamic_mask_dir/self.image_names[index])).float() / 255.).to(device).unsqueeze(0)

    def load_dynamic_masks(self, indices):
        return torch.cat([self.load_dynamic_mask(i) for i in indices], 0)

    ################### RGB images ####################
    def extract_frames(self):
        if os.path.isdir(self.input_video): # a directory of frames
            frame_dir = Path(self.input_video)
            self.image_dir['full'].makedirs_p()
            os.system('cp {} {}'.format(frame_dir/'*.{}'.format(self.opt.image_ext), self.image_dir['full']))
        else: # a video file
            if not self.preprocess: return
            self.image_dir['full'].makedirs_p()
            os.system('ffmpeg -y -hide_banner -loglevel panic -i "{}" {}/%05d.{}'.format(
                self.input_video, self.image_dir['full'], self.opt.image_ext))

    def downsample_frames(self):

        self.image_dir['flow'].makedirs_p()
        self.image_dir['down'].makedirs_p()
        self.image_dir['key'].makedirs_p()
        
        if self.image_size['full'][1] > self.image_size['full'][0]:
            a = self.image_size['full'][1]
            b = self.image_size['full'][0]
        else:
            a = self.image_size['full'][0]
            b = self.image_size['full'][1]
    
        a_depth = self.opt.depth_size
        if a >= 1024:
            a_flow = 1024
        else:
            a_flow = int(np.round(a / 64) * 64)

        b_flow = int(np.round(b * a_flow / a / 64) * 64)
        b_depth = int(np.round(b * a_depth / a / 32) * 32)
        
        if self.image_size['full'][1] > self.image_size['full'][0]:
            self.image_size['flow'] = (b_flow, a_flow)
            self.image_size['down'] = (b_depth, a_depth)
        else:
            self.image_size['flow'] = (a_flow, b_flow)
            self.image_size['down'] = (a_depth, b_depth)

        print_text('Downsample INFO\nOriginal: {}\nFor flow: {}\nFor depth: {}'.format(
            self.image_size['full'], self.image_size['flow'], self.image_size['down']),
            'H2', verbose=2, box=True)
        
        if not self.preprocess: return
        
        with Pool(processes=self.opt.num_workers) as pool:
            return pool.map(self.downsample_frame, list(range(self.length)))

    def downsample_frame(self, index):
        image_name = self.image_names[index]
        
        image = imread(self.image_dir['full']/image_name)

        image_flow = imresize(image, self.image_size['flow'])
        image_down = imresize(image, self.image_size['down'])

        imwrite(self.image_dir['flow']/image_name, (image_flow*255.).astype(np.uint8))
        imwrite(self.image_dir['down']/image_name, (image_down*255.).astype(np.uint8))

    def load_image(self, index, load_folder, load_size='down'):
        image = imread(self.image_dir[load_folder]/self.image_names[index])
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.
        if load_size != load_folder:
            image = F.interpolate(image, self.image_size[load_size], mode='area')
        return image

    def load_images(self, indices, load_size='down'):
        return torch.cat([self.load_image(i, load_size) for i in indices], 0)

    def save_image(self, filename, image):
        filename = self.root/filename
        if len(image.shape) == 4:
            B, C, H, W = image.shape
            image = image.permute(1, 2, 0, 3).reshape(C, H, -1)
        image = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        imwrite(filename, image)
    

    ################### Depth ####################
    def load_depth(self, index, load_size='down', load_subdir=None):
        if load_subdir is None:
            depth_filename = self.depth_dir/self.image_names[index][:-4] + '.npy'
        else:
            depth_filename = self.depth_dir/load_subdir/self.image_names[index][:-4] + '.npy'
        try:
            depth = np.load(depth_filename)
        except FileNotFoundError as e:
            depth = np.zeros([1, self.image_size['down'][0], self.image_size['down'][1]])
        depth = torch.from_numpy(depth).float().to(device)
        if load_size != 'down':
            depth = F.interpolate(depth, self.image_size[load_size], mode='area')
        return depth

    def load_depths(self, indices, load_size='down', load_subdir=None):
        depths = []
        for i in indices:
            depths.append(self.load_depth(i, load_size='down', load_subdir=load_subdir))
        depths = torch.stack(depths, 0)
        if load_size != 'down':
            depths = F.interpolate(depths, self.image_size[load_size], mode='area')
        return depths

    def load_target_depth(self, index, load_size='down'):
        depth = imread(self.gt_depth_paths[index])
        depth = np.asarray(depth).astype(np.float32)
        depth = 1 / (depth + 1e-6)
        if self.target_depth_ext[-4:] == '.pfm':
            depth = depth[..., ::-1, :]
        elif self.target_depth_ext[-4:] == '.png':
            depth *= 500
        depth = torch.Tensor(depth.copy()).float().to(device).unsqueeze(0)
        return depth

    def load_target_depths(self, indices, load_size='down'):
        self.gt_depth_paths = sorted(list(glob.glob(self.target_depth_dir/self.target_depth_ext)))
        depths = []
        for i in indices:
            depth = self.load_target_depth(i, load_size=load_size)
            depths.append(depth)
        
        depths = torch.stack(depths, 0)
        depths = F.interpolate(depths, self.image_size[load_size], mode='area')
        return depths 

    def generate_target_depth(self):
        if not self.preprocess: return
        self.target_depth_ext = '*.pfm'
        self.target_depth_dir.makedirs_p()
        p = os.getcwd()
        os.chdir('MiDaS')
        os.system('python3 run.py --model_type midas_v21 -i {} -o {} -m {}'.format(
            Path(p)/self.image_dir['down'], 
            Path(p)/self.target_depth_dir, 
            Path(p)/midas_weight_path))
        os.chdir(p)

    ################### Camera ####################
    def load_camera(self, index, load_size='down', load_subdir=None):
        if load_subdir is None:
            camera_filename = self.camera_dir/self.image_names[index][:-4] + '.npy'
        else:
            camera_filename = self.camera_dir/load_subdir/self.image_names[index][:-4] + '.npy'
        try:
            camera = np.load(camera_filename, allow_pickle=True)[()]
            K = torch.from_numpy(camera['K']).float().to(device)
            pose = torch.from_numpy(camera['pose']).float().to(device)
        except FileNotFoundError as e:
            K = torch.eye(3).float().to(device)
            pose = torch.eye(4).float().to(device)
        return K, pose

    def load_cameras(self, indices, load_subdir=None):
        Ks, poses = [], []
        for i in indices:
            K, pose = self.load_camera(i, load_subdir=load_subdir)
            Ks.append(K)
            poses.append(pose)
        Ks = torch.stack(Ks, 0)
        poses = torch.stack(poses, 0)
        Ks_inv = inverse_intrinsic(Ks)
        poses_inv = inverse_pose(poses)
        return Ks, Ks_inv, poses, poses_inv


    ################### General ####################
    def load_video_info(self):
        self.image_names = sorted(list(glob.glob(self.image_dir['full']/'*.{}'.format(self.opt.image_ext))))
        self.image_names = [os.path.split(image_name)[-1] for image_name in self.image_names]
        self.length = len(self.image_names)
        
        # get frame size
        sample_image = imread(self.image_dir['full']/self.image_names[0])
        self.image_size = {'full': sample_image.shape[:2]}
        
        if os.path.isdir(self.input_video):
            self.fps = 30
        else: 
            # get fps
            p = subprocess.check_output(['ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}'.format(self.input_video)], shell=True)
            exec('self.fps = float({})'.format(p.decode('utf-8').rstrip('\n')))
            
        print_text('Video INFO\nPath: {}\nNumber of frames: {}\nFrame rate: {}\nFrame size: {}'.format(
            self.input_video, self.length, self.fps, self.image_size['full']),
            'H2', verbose=2, box=True)

    @torch.no_grad()
    def get_items(self, 
            indices, 
            load_size='down', 
            load_depth=False, 
            load_camera=False, 
            load_flow=True, 
            segment=False, 
            keyframe=False, 
            load_subdir=None,
            flow_intervals=[1]
        ):

        h, w = self.image_size[load_size]
        items = {}
        indices = list(indices)

        # image
        items['image'] = self.load_images(indices, load_size)
        
        # dynamic mask
        items['dyn_mask'] = self.load_dynamic_masks(indices)
        if items['dyn_mask'].shape[-1] != w or items['dyn_mask'].shape[-2] != h:
            items['dyn_mask'] = F.interpolate(items['dyn_mask'].unsqueeze(1), (h, w), mode='area')
        
        # depth
        if load_depth:
            items['depth'] = self.load_depths(indices, load_size=load_size, load_subdir=load_subdir)
            items['gt_depth'] = self.load_target_depths(indices, load_size=load_size)
        
        # camera
        if load_camera:
            items['K'], items['K_inv'], items['pose'], items['pose_inv'] = self.load_cameras(indices, load_subdir=load_subdir)

        # flow
        if load_flow:
            for i in flow_intervals:
                flow_f, flow_b, flow_fm, flow_bm = self.load_sequence_flows(
                        indices, 
                        load_size=load_size, 
                        interval=i, 
                        folder='key' if keyframe else 'seq')

                items[('flow_f', i)] = flow_f
                items[('flow_b', i)] = flow_b
                items[('flow_f_mask', i)] = flow_fm
                items[('flow_b_mask', i)] = flow_bm
                items[('flow_f_mask', i)] = torch.ones(items[('flow_f', i)].shape[:-1]).unsqueeze(1).to(device)
                items[('flow_b_mask', i)] = torch.ones(items[('flow_b', i)].shape[:-1]).unsqueeze(1).to(device)

        return items 

    def save_items(self, indices, items, save_subdir=None):
        
        self.depth_dir.makedirs_p()
        self.camera_dir.makedirs_p()
        
        self.depth_dir_to_save = self.depth_dir/save_subdir if save_subdir is not None else self.depth_dir
        self.camera_dir_to_save = self.camera_dir/save_subdir if save_subdir is not None else self.camera_dir
        self.depth_dir_to_save.makedirs_p()
        self.camera_dir_to_save.makedirs_p()
        
        self.items_to_save = {}
        for k in ['depth', 'K', 'pose']:
            if k in items.keys():
                self.items_to_save[k] = items[k].detach().cpu().numpy()

        with Pool(processes=self.opt.num_workers) as pool:
            return pool.map(self.save_item, enumerate(indices))
            

    def save_item(self, index, save_subdir=None):
        i, index = index
        filename = self.image_names[index]
        items = self.items_to_save
        if 'depth' in items.keys():
            # depth
            depth = items['depth'][i]
            np.save(self.depth_dir_to_save/filename[:-4] + '.npy', depth)
        
        if 'K' in items.keys() and 'pose' in items.keys():
            # camera
            K =    items['K'][i]
            pose = items['pose'][i]
            np.save(self.camera_dir_to_save/filename[:-4] + '.npy', {'K': K, 'pose': pose})

if __name__ == '__main__':
    import options
    opt = options.Options().parse()

    seq_io = SequenceIO(opt, False)
