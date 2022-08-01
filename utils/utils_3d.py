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


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.002):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
        points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
        lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
        colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
        radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)
    
    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                #cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)
            
            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


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

    #cam_set = LineMesh(cam_plane_pts, lines, colors)
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
        mask = depth > -1
    
    mask = mask.reshape(-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cam_pts[mask > 0, :])
    if color is None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3)[mask > 0, :])
    else:
        pcd.colors = o3d.utility.Vector3dVector([color for i in range(int(mask.sum()))])
    #pcd = pcd.voxel_down_sample(voxel_size=0.02)
    return pcd

def add_cams(poses, cam_color, line_color, visualizer, K, show_cam=False, cam_interval=1):
    import visualization_tools.visualize_3d as visualize_3d
    import open3d as o3d
    cam_poses = []
    for i, pose in enumerate(poses):
        if pose is not None:
            R, T = pose[:3, :3], pose[:3, 3]
            cam, cam_pose = visualize_3d.gen_cam_lineset((480, 640), K, color=cam_color, 
                                        R=R, T=T, size_scale=0.02)
            cam_poses.append(cam_pose)
            if show_cam and (i % cam_interval == 0 or i == poses.shape[0]-1):
                visualizer.add_geometry(cam)


    cam_poses = np.asarray(cam_poses)
    line = LineMesh(cam_poses, [[i,i+1] for i in range(cam_poses.shape[0]-1)], line_color)
    line.add_line(visualizer)
    '''
    traj_set = o3d.geometry.LineSet()
    traj_set.points = o3d.utility.Vector3dVector(cam_poses)
    traj_set.lines = o3d.utility.Vector2iVector(
                            [[i, i+1] for i in range(cam_poses.shape[0] - 1)])
    traj_set.colors = o3d.utility.Vector3dVector(
                            [line_color for i in range(cam_poses.shape[0] - 1)])
    visualizer.add_geometry(traj_set)
    '''

def add_point_cloud(image, pts2d, pts3d, visualizer):
    
    import open3d as o3d
    image /= 255.

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pts2d_ = np.round(pts2d).astype(np.int32)
    pcd.colors = o3d.utility.Vector3dVector(image[pts2d_[:,1], pts2d_[:,0]])

    visualizer.add_geometry(pcd)

