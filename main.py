import os, sys, math
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R

from sequence_io import SequenceIO
from models import *
from sfm import SFM
from pose_graph import PoseGraph
from project_3d import inverse_pose
from depth_filter import DepthFilter
import options
from logger import print_text, progress_bar 

opt = options.Options().parse()
global device
device = torch.device(opt.cuda)

class VideoDepthOptimizer:
    def __init__(self, preprocess=False):

        print_text('Preprocess video', 'H1', verbose=1)
        self.seq_io = SequenceIO(opt, preprocess)
        
        print_text('Load models', 'H1', verbose=1)
        self.models = {
            'depth': get_depth_model(opt, self.seq_io).to(device),
            'pose':  get_pose_model(opt, self.seq_io).to(device)
        }
        if opt.mesh_deformation:
            self.models['mesh'] = get_mesh_model(opt, self.seq_io).to(device)

        self.sfm = SFM(self.models, self.seq_io, opt)

        self.pose_graph = PoseGraph(opt, self.models['pose'], self.seq_io) 

        self.keys = self.seq_io.keyframes
        self.segment_load_dir = None 
        
    def process_keyframes(self):

        print_text('Reconstruct keyframes', 'H1', verbose=1)
        i = 0 

        keyframe_groups = []

        overlap_length = max(opt.keyframe_intervals)
        
        step_size = opt.keyframe_max_batch_size
        
        num_steps = 1 + int(np.ceil((len(self.keys) - step_size + 1) / (step_size - overlap_length + 1)))
        pbar = progress_bar(opt.num_epoch_keyframe * num_steps)

        while i < len(self.keys) - 1:
            j = min(i + step_size, len(self.keys) - 1)
            if i > 0 and i + overlap_length >= j: break
            
            indices = self.keys[i:j+1]
            
            keyframes = self.seq_io.get_items(indices, 
                keyframe=True, load_depth=True, load_camera=True, flow_intervals=[1])
            keyframe_groups.append(indices)
            L = len(indices)

            trainable_indices = list(range(L))
            if i > 0: trainable_indices = trainable_indices[overlap_length:]
            
            loss_pairs = []
            for k in opt.keyframe_intervals:
                if k >= L: continue
                loss_pairs.append({
                    'mode': 'seq_flow' if k == 1 else 'no_flow',
                    'interval': k,
                    'weight': k ** -1,
                    'grad': False
                })
            keyframes = self.sfm.reconstruct_images(
                keyframes,
                indices,
                trainable_indices,
                trainable_indices,
                loss_pairs,
                opt.num_epoch_keyframe,
                opt.keyframe_lr,
                opt.keyframe_scales,
                pose_base=True,
                log_name='key-{}'.format(i),
                pbar=pbar
                )
            
            self.seq_io.save_items(indices, keyframes)
            self.seq_io.save_items(indices, keyframes, save_subdir='pre-optimize')
            self.sfm.save_weights('keyframe-{}-{}'.format(self.keys[i], self.keys[j]))

            if j - overlap_length <= i: break
            i = j - overlap_length

        self.sfm.save_weights('keyframe')
        np.save(self.seq_io.root/'keyframe_groups.npy', keyframe_groups)
        self.segment_load_dir = None 
        pbar.close()

    
    def pose_graph_optimization(self):
        print_text('Pose graph optimization', 'H1', verbose=1)
         
        # compute pose graph
        self.sfm.load_weights('keyframe')
        edges = self.pose_graph.compute_graph()
        
        batch_i = 0
        while batch_i < len(edges):
            batch_i_end = min(batch_i + opt.segment_max_batch_size // 2, len(edges))

            batch_edges = edges[batch_i:batch_i_end]

            edge_items = {}
            loss_pairs = []
            for i, edge in enumerate(batch_edges):
                edge_item = self.seq_io.get_items(edge, load_depth=True, load_camera=True, load_flow=False)
                flow_f, flow_b, mask_f, mask_b = self.seq_io.load_graph_flow(edge[0], edge[1])
                
                edge_item[('flow_f', 1)] = flow_f
                edge_item[('flow_b', 1)] = flow_b
                edge_item[('flow_f_mask', 1)] = mask_f
                edge_item[('flow_b_mask', 1)] = mask_b
                for k, v in edge_item.items():
                    if k not in edge_items.keys():
                        edge_items[k] = v
                    else:
                        edge_items[k] = torch.cat((edge_items[k], v), 0)

            
            pbar = progress_bar(opt.num_epoch_edge)
            indices = np.array(batch_edges).reshape(-1)
            L = len(indices) 
            loss_pairs.append({
                'mode': 'seq_flow',
                'a': list(range(0, L, 2)),
                'b': list(range(1, L, 2)),
                'weight': 1,
                'interval': 1,
                'grad': False
            })
            
            items = self.sfm.reconstruct_images(
                edge_items,
                indices,
                list(range(L)),
                list(range(L)),
                loss_pairs,
                opt.num_epoch_edge,
                opt.segment_lr,
                opt.segment_scales,
                pose_base=False,
                log_name='edge-{}'.format(batch_i),
                pbar=pbar
                )

            self.seq_io.save_items(indices, edge_items, save_subdir='graph')
            pbar.close()
    
            batch_i = batch_i_end
        
        # pose graph optimization
        self.pose_graph.optimize()
        self.segment_load_dir = 'optimize' 

    def process_segments(self):
        print_text('Reconstruct segments', 'H1', verbose=1)

        keyframe_groups = np.load(self.seq_io.root/'keyframe_groups.npy', allow_pickle=True)[()]
        
        segments = []
        for group_index, group in enumerate(keyframe_groups):
            i = 1
            anchor_l = group[0]
            while anchor_l < group[-1]:
                while i < len(group) and group[i] - anchor_l < opt.segment_max_batch_size:
                    i += 1
                i -= 1
                anchor_r = group[i]
                assert anchor_l != anchor_r
                segments.append([anchor_l, anchor_r, group_index])
                anchor_l = anchor_r

        pbar = progress_bar(len(segments) * opt.num_epoch_segment)

        for i in range(len(segments)):
            anchor_l, anchor_r, group_index = segments[i]

            segment_indices = np.array(list(range(anchor_l, anchor_r + 1)))
            segment = self.seq_io.get_items(segment_indices, load_depth=True, load_camera=True, load_subdir=self.segment_load_dir)
            
            # load model
            group_l = keyframe_groups[group_index][0]
            group_r = keyframe_groups[group_index][-1]
            self.sfm.load_weights('keyframe-{}-{}'.format(group_l, group_r))


            # decide loss pairs
            L = len(segment_indices)

            loss_pairs = []
            for k in opt.segment_intervals:
                if k >= L: continue
                loss_pairs.append({
                    'mode': 'seq_flow' if k == 1 else 'no_flow',
                    'interval': k,
                    'weight': k ** 0.5,
                    'grad': True
                })


            non_key_indices = []
            key_indices = []
            for l in range(L):
                if segment_indices[l] in self.keys: key_indices.append(l)
                else: non_key_indices.append(l)
             
            segment = self.sfm.reconstruct_images(
                segment, 
                segment_indices, 
                list(range(L)),
                non_key_indices,
                loss_pairs,
                opt.num_epoch_segment,
                opt.segment_lr,
                opt.segment_scales,
                pose_base=True,
                log_name='segment-{}-{}'.format(anchor_l, anchor_r),
                pbar=pbar
            )
            self.seq_io.save_items(segment_indices, segment, save_subdir=opt.save_subdir)
    
    def post_filter(self):
        print_text('Post processing with depth filter', 'H1', verbose=1)
        df = DepthFilter(opt, self.seq_io)
        df.process_sequence()


vd = VideoDepthOptimizer(not opt.skip_preprocess)
if not opt.skip_keyframe:
    vd.process_keyframes()
    if opt.pose_graph:
        vd.pose_graph_optimization()

if opt.pose_graph:
    vd.segment_load_dir='optimize'

if not opt.skip_segment:
    vd.process_segments()

if opt.post_filter:
    vd.post_filter()
