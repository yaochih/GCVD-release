import open3d as o3d
import numpy as np
import cv2 as cv
from path import Path
from imageio import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from imageio import imwrite
import os, sys, threading, time, argparse, glob

from utils_vis import *


class GCVDReader():
    def __init__(self, folder):
        self.depth_dir = Path(folder)/'depths/final'
        self.camera_dir = Path(folder)/'camera/final'
        self.image_dir = Path(folder)/'images_down'

        self.image_filenames = sorted(list(glob.glob(self.image_dir/'*')))
        self.depth_filenames = sorted(list(glob.glob(self.depth_dir/'*')))
        self.camera_filenames = sorted(list(glob.glob(self.camera_dir/'*')))
        
        self.length = len(self.image_filenames)
        assert len(self.depth_filenames) == self.length
        assert len(self.camera_filenames) == self.length

    def read_camera(self, index):
        camera = np.load(self.camera_filenames[index], allow_pickle=True)[()]
        K = camera['K']
        pose = camera['pose']
        return K, pose

    def read(self, index):
        image = imread(self.image_filenames[index]).astype(float) / 255.
        depth = np.load(self.depth_filenames[index]).squeeze(0)
        K, pose = self.read_camera(index)
        
        return image, depth, K, pose

class SevenScenesReader():
    def __init__(self, folder):
        self.camera_filenames = sorted(list(glob.glob(Path(folder)/'*.pose.txt')))
        self.length = len(self.camera_filenames)

    def read_camera(self, index):
        pose = np.genfromtxt(self.camera_filenames[index])
        return None, pose

def read_trajectory(reader):
    poses = []
    for i in range(reader.length):
        poses.append(reader.read_camera(i)[1])
    poses = np.stack(poses, axis=0)
    return poses

index = 0
parser = argparse.ArgumentParser(description='3D Visualization')
parser.add_argument('input_path', type=str)
parser.add_argument('groundtruth_path', type=str)
opt = parser.parse_args()

reader = GCVDReader(opt.input_path)
gt_reader = SevenScenesReader(opt.groundtruth_path)
gcvd_traj = read_trajectory(reader)
gt_traj = read_trajectory(gt_reader)

gcvd_traj_aligned, align_transform, ate = align_trajectory(gt_traj, gcvd_traj)
print('ATE = {:.04} m'.format(ate))

app = o3d.visualization.gui.Application.instance
app.initialize()

height, width = 1080, 1920
window = o3d.visualization.gui.Application.instance.create_window('img', width=width, height=height)
widget = o3d.visualization.gui.SceneWidget()
widget.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
vis = widget.scene

window.add_child(widget)
mat = o3d.visualization.rendering.Material()
mat.shader = 'unlitLine'
mat.line_width = 3

mat_g = o3d.visualization.rendering.Material()
mat_g.shader = 'unlitLine'
mat_g.line_width = 1.5

mat_pcd = o3d.visualization.rendering.Material()

view_cam = widget.scene.camera

# view offset
offset = np.array([[ 9.99663439e-01,  2.59491099e-02,  1.59831567e-04, -5.18736794e-02],
                   [ 2.12940823e-02, -8.16779167e-01, -5.76557411e-01, -1.95576986e+00],
                   [-1.48307187e-02,  5.76366692e-01, -8.17056646e-01, -1.99995985e+00],
                   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

prev_pose = None

def get_look_at_param(V=None):
    if V is None:
        global view_cam
        V = view_cam.get_view_matrix()
        V = np.linalg.inv(V)
    eye = V[:3, 3]
    Z = V[:3, 2]
    Y = V[:3, 1]
    X = V[:3, 0]
    center = eye - Z
    up = np.cross(Z, X) / np.dot(Z, Z) + Z
    return center, eye, up 

def add_new_cam():
    global index, reader, vis, view_cam, widget, prev_pose, offset, align_transform, gcvd_traj_aligned, gt_traj

    if index >= reader.length: return 
    image, depth, K, pose = reader.read(index)
   
    align_R, align_t, align_s = align_transform
    rot_aligned = align_R @ pose[:3, :3]

    try:
        view_param = get_look_at_param(gt_traj[index] @ offset)
        view_cam.look_at(list(view_param[0]), list(view_param[1]), list(view_param[2]))

    except Exception as e:
        pass

    R = rot_aligned
    t = gcvd_traj_aligned[index]
    
    cam_set, _ = gen_cam_lineset(image.shape[:2], K, color=[1, 0, 0], R=R, T=t, size_scale=0.2)
    vis.remove_geometry('cam_{}'.format(index-1))
    vis.add_geometry('cam_{}'.format(index), cam_set, mat)

    pcd = compute_point_cloud(image, depth * align_s, K, R, t, color=None)
    vis.remove_geometry('cloud')
    vis.add_geometry('cloud'.format(index), pcd, mat_pcd)

 
    if index > 0:
        traj_lineset = o3d.geometry.LineSet()
        traj_lineset.colors = o3d.utility.Vector3dVector([[0, 1, 0]])
        traj_lineset.points = o3d.utility.Vector3dVector(gcvd_traj_aligned[index - 1: index + 1])
        traj_lineset.lines = o3d.utility.Vector2iVector([[0, 1]])
        vis.add_geometry('traj_{}'.format(index), traj_lineset, mat)
    
    index += 1

gt_lineset = o3d.geometry.LineSet()
gt_lineset.points = o3d.utility.Vector3dVector(gt_traj[:, :3, 3])
gt_lineset.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(gt_reader.length - 1)])
gt_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1] for i in range(gt_reader.length - 1)])
vis.add_geometry('gt_traj', gt_lineset, mat_g)

def thread_main():
    while True:
        o3d.visualization.gui.Application.instance.post_to_main_thread(window, add_new_cam)
        time.sleep(0.03)

vis_thread = threading.Thread(target=thread_main)
vis_thread.start()
o3d.visualization.gui.Application.instance.run()

