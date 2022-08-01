from .depth import *
from .pose import *
from .mesh_deformer import MeshDeformer

def get_mesh_model(opt, seq_io):
    return MeshDeformer(opt, seq_io)

def get_depth_model(opt, seq_io, model=None):
    model = model if model is not None else opt.depth_model
    if model == 'midas':
        return DepthMidas(opt, seq_io)
    elif model == 'midas_multiscale':
        return DepthMidasMultiScale(opt, seq_io)
    elif model == 'resnet':
        return DepthResnet(opt, seq_io)
    else:
        raise NotImplementedError

def get_pose_model(opt, seq_io, model=None):
    model = model if model is not None else opt.pose_model
    if model == 'resnet':
        return PoseResnet(opt, seq_io)
    else:
        raise NotImplementedError

