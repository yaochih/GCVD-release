from __future__ import absolute_import, division, print_function

import os, argparse

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='video depth')

        self.parser.add_argument('input_path',
                                 type=str)
        self.parser.add_argument('--output_dir',
                                 type=str,
                                 default='outputs')
        self.parser.add_argument('--name',
                                 type=str,
                                 default='test')
        
        # FEATURE
        self.parser.add_argument('--positional_encoding',
                                 action='store_true',
                                 default=True)
        self.parser.add_argument('--mesh_deformation',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--pose_graph',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--post_filter',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--depth_size',
                                 type=int,
                                 default=384)
        self.parser.add_argument('--mesh_size',
                                 type=int,
                                 default=16)
        self.parser.add_argument('--keyframe_thresh',
                                 type=float,
                                 default=0.1)

        # HYPERPARAMETER FOR OPTIMIZATION
        self.parser.add_argument('--num_epoch_keyframe',
                                 type=int,
                                 default=300)
        self.parser.add_argument('--num_epoch_edge',
                                 type=int,
                                 default=300)
        self.parser.add_argument('--num_epoch_segment',
                                 type=int,
                                 default=100)
        self.parser.add_argument('--keyframe_scales',
                                 nargs='+',
                                 type=int,
                                 default=[2])
        self.parser.add_argument('--segment_scales',
                                 nargs='+',
                                 type=int,
                                 default=[0])
        self.parser.add_argument('--keyframe_intervals',
                                 nargs='+',
                                 default=[1, 2, 4, 8])
        self.parser.add_argument('--segment_intervals',
                                 nargs='+',
                                 default=[1, 2, 4, 8])
        self.parser.add_argument('--keyframe_lr',
                                 type=float,
                                 default=2e-4)
        self.parser.add_argument('--edge_lr',
                                 type=float,
                                 default=5e-5)
        self.parser.add_argument('--segment_lr',
                                 type=float,
                                 default=1e-4)
        self.parser.add_argument('--loss_photo',
                                 type=float,
                                 default=1.0)
        self.parser.add_argument('--loss_flow',
                                 type=float,
                                 default=10.0)
        self.parser.add_argument('--loss_depth',
                                 type=float,
                                 default=0.5)
        self.parser.add_argument('--loss_smooth',
                                 type=float,
                                 default=0)
        self.parser.add_argument('--loss_mesh',
                                 type=float,
                                 default=0.5)
        self.parser.add_argument('--loss_depth_grad',
                                 type=float,
                                 default=0.1)
        self.parser.add_argument('--ssim_weight',
                                 type=float,
                                 default=0.5)
        self.parser.add_argument('--loss_depth_mode',
                                 choices=['ratio', 'minmax'],
                                 default='ratio')
        self.parser.add_argument('--camera_scale',
                                 type=float,
                                 default=300)

        # MODEL
        self.parser.add_argument('--depth_model',
                                 choices=['midas', 'midas_multiscale', 'resnet'],
                                 default='midas')
        self.parser.add_argument('--pose_model',
                                 choices=['resnet'],
                                 default='resnet')
        self.parser.add_argument('--target_depth',
                                 choices=[None, 'midas', 'boost'],
                                 default='midas')
        
        # OPTIMIZATION SETTING
        self.parser.add_argument('--optimizer',
                                 choices=['Adam', 'RAdam'],
                                 default='Adam')
        self.parser.add_argument('--learn_intrinsic',
                                 action='store_true',
                                 default=False)

        # VISUALIZATION (for demo_3d.py)
        self.parser.add_argument('--viz_cam',
                                 default='all',
                                 choices=['all', 'keyframe', 'none'])
        self.parser.add_argument('--viz_depth',
                                 default='none',
                                 choices=['all', 'keyframe', 'none'])
        self.parser.add_argument('--viz_dir',
                                 default='final',
                                 choices=['none', 'optimize', 'final', 'pre-optimize'])

        # SYSTEM
        self.parser.add_argument('--skip_preprocess',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--regenerate_keyframe',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--skip_keyframe',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--skip_segment',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--save_subdir',
                                 default='final')
        self.parser.add_argument('--cuda',
                                 type=str,
                                 default='cuda')
        self.parser.add_argument('--keyframe_max_batch_size',
                                 type=int,
                                 default=40)
        self.parser.add_argument('--segment_max_batch_size',
                                 type=int,
                                 default=40)
        self.parser.add_argument('--image_ext',
                                 default='png')
        self.parser.add_argument('--log_tensorboard',
                                 action='store_true',
                                 default=False)
        self.parser.add_argument('--log_freq',
                                 type=int,
                                 default=1)
        self.parser.add_argument('--log_image_scale',
                                 type=float,
                                 default=0.1)
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=6)
        self.parser.add_argument('--verbose',
                                 type=int,
                                 default=3,
                                 choices=[0, 1, 2, 3, 4])


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
