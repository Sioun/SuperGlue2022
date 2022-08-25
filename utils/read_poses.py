import numpy as np
from scipy.spatial.transform import Rotation as R


def get_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    """
    locations = []
    rotations = []
    loc_reader = open(root + '\SavedPosition_' + scene + '.txt', 'r')
    rot_reader = open(root + '\SavedRotationQuaternion_' + scene + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))

    locations = np.array(locations)
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_dcm()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

    return poses, np.array(poses_mat)


def get_relative_pose(pose_t0, pose_t1):
    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)
