import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import g2o
from scipy.spatial.transform import Rotation
import cv2 as cv
from tqdm import tqdm

from flow import FlowProcessor
from logger import print_text
from utils.utils import *

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            v = self.vertex(int(v))
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()


class PoseGraph:
    def __init__(self, opt, pose_model, seq_io):
        self.pose_encoder = pose_model.encoder
        self.feature_base = []
        self.opt = opt
        self.seq_io = seq_io

    def extract_feature(self, image):
        feature = self.pose_encoder(image)
        avg_feature = F.avg_pool2d(feature, feature.shape[-2:])
        avg_feature = avg_feature.view(512, -1).transpose(1, 0)
        feature = F.normalize(avg_feature, dim=-1)
        return feature
    
    @torch.no_grad()
    def compute_graph(self, 
            similarity_thresh=0.9,
            inlier_thresh=0.5,
            ):

        print_text('Compute pose graph', level='H2', verbose=2)
        keys = self.seq_io.keyframes
        for i in keys:
            image = self.seq_io.load_image(i, 'key')
            self.feature_base.append(self.extract_feature(image).detach())

        self.feature_base = torch.stack(self.feature_base, 0)

        num_keys = len(keys)
        self.graph = torch.zeros(num_keys, num_keys)

        for i in range(1, num_keys):
            query = self.feature_base[i:i+1]
            key = self.feature_base[:i]
            value = query @ key.permute(0, 2, 1)
            value = value.view(i, -1)
            self.graph[i, :i] = value.max(-1)[0].detach().cpu()
        
        self.graph[self.graph < similarity_thresh] = 0
        nms_step = max(self.opt.keyframe_intervals) 
        
        pool_layer = nn.MaxPool2d(nms_step, stride=nms_step, return_indices=True, ceil_mode=True)
        unpool_layer = nn.MaxUnpool2d(nms_step, stride=nms_step)

        pool_graph, pool_indices = pool_layer(self.graph.unsqueeze(0).unsqueeze(0))
        unpool_graph = unpool_layer(pool_graph, pool_indices, output_size=self.graph.unsqueeze(0).unsqueeze(0).shape)
        unpool_graph = unpool_graph.squeeze(0).squeeze(0)
        unpool_graph[unpool_graph < similarity_thresh] = 0

        edges = unpool_graph.nonzero(as_tuple=False)
        
        self.seq_io.flow_dir['graph'].makedirs_p()
        flow = FlowProcessor(self.opt).to(torch.device(self.opt.cuda))
        verified_edges = []
        
        shift_threshold = self.seq_io.shift_thresh
        
        for t, r in tqdm(edges):
            if abs(t - r) < nms_step: continue
            tgt, ref = keys[t], keys[r]

            img_tgt = self.seq_io.load_image(tgt, 'key', load_size='flow')
            img_ref = self.seq_io.load_image(ref, 'key', load_size='flow')
            flow_f, flow_b, mask_f, mask_b, inlier_ratios = flow.get_flow_forward_backward(img_tgt, img_ref, pre_homo=True)
           
            flow_mag = torch.sqrt(torch.sum(torch.pow(flow_f - get_grid(flow_f), 2), 1))
            if len(inlier_ratios) == 0 or inlier_ratios[0] < inlier_thresh: continue
            if mask_f.mean() < 0.1: continue
            if flow_mag.mean() > shift_threshold: continue
            
            verified_edges.append([tgt, ref])

            np.save(self.seq_io.flow_dir['graph']/'{}_{}.npy'.format(tgt, ref),
                    {'flow_f': flow_f, 'flow_b': flow_b, 'mask_f': mask_f, 'mask_b': mask_b})
        
        np.save(self.seq_io.root/'pose_graph_edges.npy', np.array(verified_edges))
        
        self.edges = verified_edges
        print_text('Verified graph: ' + str(self.edges), level='H2', verbose=3)
        
        del flow

        return self.edges

    def visualize_graph(self, data):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        ax = plt.subplot()
        im = ax.imshow(data)#, vmin=0, vmax=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.show()

    def optimize(self):
        self.edges = np.load(self.seq_io.root/'pose_graph_edges.npy')
        optimizer = PoseGraphOptimization()

        keys = list(self.seq_io.keyframes)
        
        items = {}
        items['K'], items['K_inv'], items['pose'], items['pose_inv'] = self.seq_io.load_cameras(keys)
        
        self.seq_io.save_items(keys, items, save_subdir='pre-optimize')
        seq_P = items['pose'].cpu().numpy()

        for i, key in enumerate(keys):
            pose = g2o.Isometry3d(seq_P[i, :3, :3], seq_P[i, :3, 3])
            optimizer.add_vertex(key, pose, fixed=True if i == 0 else False)
        
        for i in self.opt.keyframe_intervals:
            for k1, k2 in zip(keys[:-i], keys[i:]):
                i1, i2 = keys.index(k1), keys.index(k2)
                rel_P = (items['pose_inv'][i1] @ items['pose'][i2]).detach().cpu().numpy()
                rel_pose = g2o.Isometry3d(rel_P[:3, :3], rel_P[:3, 3])
                optimizer.add_edge([k1, k2], rel_pose, information=np.identity(6) / i)

        for edge in self.edges:
            K, K_inv, P, P_inv = self.seq_io.load_cameras(edge, load_subdir='graph')
            rel_P = (P_inv[0] @ P[1]).detach().cpu().numpy()
            rel_pose = g2o.Isometry3d(rel_P)
            optimizer.add_edge(edge, rel_pose)
        
        if self.opt.verbose >= 3:
            optimizer.set_verbose(True)
        optimizer.optimize(100)

        for i, key in enumerate(keys):
            pose = optimizer.get_pose(key)
            items['pose'][i, :3, :3] = torch.Tensor(pose.R).float().to(items['pose'].device) 
            items['pose'][i, :3, 3]  = torch.Tensor(pose.t).float().to(items['pose'].device) 
        
        self.seq_io.save_items(keys, items, save_subdir='optimize')

