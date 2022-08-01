import numpy as np
import open3d as o3d


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


def gen_cam_lineset(img_size, K, color=[1, 0, 0], R=None, T=None, size_scale=1.):
    h, w = img_size
    img_plane_pts = np.zeros((8, 3)).astype(np.float64)
    img_plane_pts[1, 0] = w-1.
    img_plane_pts[2, 0] = w-1.
    img_plane_pts[2, 1] = h-1.
    img_plane_pts[3, 1] = h-1.
    img_plane_pts[:4, 2] = 1.
    img_plane_pts *= 0.5 * size_scale
    
    cam_plane_pts = (np.linalg.inv(K) @ img_plane_pts.transpose()).transpose()
    '''
    cam_plane_pts[5, 0] = 1. #X axis
    cam_plane_pts[6, 1] = 1. #Y axis
    cam_plane_pts[7, 2] = 1. #Z axis
    cam_plane_pts[5:] *= 0.5 * size_scale
    '''
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4], [4, 5], [4, 6], [4, 7]])
    colors = [color for i in range(len(lines)-3)]
    colors += [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
     
    if R is not None and T is not None:
        cam_plane_pts = (R @ cam_plane_pts.transpose()).transpose() + T

    #'''
    cam_set = o3d.geometry.LineSet()
    cam_set.points = o3d.utility.Vector3dVector(cam_plane_pts)
    cam_set.lines = o3d.utility.Vector2iVector(lines)
    cam_set.colors = o3d.utility.Vector3dVector(colors)
    #'''
    return cam_set, cam_plane_pts[5]

def compute_point_cloud(rgb, depth, K, R=None, T=None, color=None, mask=None):
    h, w = rgb.shape[0], rgb.shape[1]

    xs = np.arange(0, w, dtype=np.float32)
    ys = np.arange(0, h, dtype=np.float32)
    ws = np.ones((1, h, w), dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    grid = np.vstack([np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0), ws])
    
    cam_pts = (np.linalg.inv(K) @ grid.reshape(3, -1)).reshape(3, h, w)
    cam_pts[0,:] *= depth
    cam_pts[1,:] *= depth
    cam_pts[2,:] *= depth

    cam_pts = cam_pts.reshape(3, -1)
    if R is not None and T is not None:
        cam_pts = (R @ cam_pts).transpose() + T
    
    if mask is None:
        mask = depth > 0 
    
    mask = mask.reshape(-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pts[mask > 0, :])
    if color is None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)[mask > 0, :])
    else:
        rgb = rgb.reshape(-1, 3)[mask > 0, :]
        #rgb = rgb * color
        alpha = 0.45
        rgb = rgb * alpha
        rgb += np.array(color) * (1-alpha)
        pcd.colors = o3d.utility.Vector3dVector([rgb.reshape(-1, 3)[i] for i in range(int(mask.sum()))])
    #pcd = pcd.voxel_down_sample(voxel_size=0.02)
    return pcd

def add_point_cloud(image, pts2d, pts3d, visualizer):
    
    import open3d as o3d
    image /= 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pts2d_ = np.round(pts2d).astype(np.int32)
    pcd.colors = o3d.utility.Vector3dVector(image[pts2d_[:,1], pts2d_[:,0]])

    visualizer.add_geometry(pcd)

def align_trajectory(traj_1, traj_2):
    pts_1 = traj_1[:, :3, 3].T
    pts_2 = traj_2[:, :3, 3].T
    traj_1_zerocenter = (pts_1.T - pts_1.mean(1)).T
    traj_2_zerocenter = (pts_2.T - pts_2.mean(1)).T

    W = np.zeros((3, 3))
    for col in range(pts_1.shape[1]):
        W += np.outer(traj_2_zerocenter[:, col], traj_1_zerocenter[:, col])

    U, d, Vh = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = np.dot(np.dot(U, S), Vh)

    traj_2_r = rot @ traj_2_zerocenter
    dots = 0.0
    norms = 0.0

    for col in range(traj_2_zerocenter.shape[1]):
        dots += np.dot(traj_1_zerocenter[:, col].T, traj_2_r[:, col])
        normi = np.linalg.norm(traj_2_zerocenter[:, col])
        norms += normi * normi

    s = float(dots / norms)
    trans = pts_1.mean(1) - s * np.dot(rot, pts_2.mean(1))

    pts_2_aligned = ((s * np.dot(rot, pts_2)).T + trans).T
    align_err = pts_2_aligned - pts_1
    align_err = np.matrix(align_err)
    trans_err = np.mean(np.sqrt(np.sum(np.multiply(align_err, align_err), 0)).A[0])
    return pts_2_aligned.T, (rot, trans, s), trans_err
