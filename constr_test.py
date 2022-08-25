import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utils.read_poses import *
from utils.geometry import *
from utils.read_depth import *
import cv2
from models.SuperPointNet_gauss2 import *

def frame2tensor(frame):
    return torch.from_numpy(frame/255.).float()[None, None]

# get data
poses_quat, poses_mat = get_poses('1', 'E:\\2021-2022 Msc\dataset\SyntheticColon_I\\')
img_frame0 = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\\Frames_S1\FrameBuffer_0009.png',cv2.IMREAD_GRAYSCALE)
img_frame1 = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\\Frames_S1\FrameBuffer_0010.png',cv2.IMREAD_GRAYSCALE)
frame0 = poses_mat[9,:,:]
frame1 = poses_mat[10,:,:]
# get 4x4 relative pos
relative_pose = get_relative_pose(frame0, frame1)
relative_pose = torch.tensor(relative_pose[0:3,:].astype(np.float32)).unsqueeze(0)

# read 3x3 intrinsics
with open('E:\\2021-2022 Msc\dataset\SyntheticColon_I\Frames_S1\cam.txt', 'r') as f:
    l = [[float(num) for num in line.split(' ')] for line in f]
intrs = torch.from_numpy(np.resize(l,(3,3)).astype(np.float32)).unsqueeze(0)
print(type(intrs))

# read depth image
depth0 = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\\Frames_S1\Depth_0009.png', -cv2.IMREAD_ANYDEPTH).astype(np.float16)
depth1 = cv2.imread('E:\\2021-2022 Msc\dataset\SyntheticColon_I\\Frames_S1\Depth_0010.png', -cv2.IMREAD_ANYDEPTH).astype(np.float16)
depth0 = torch.tensor(depth0).unsqueeze(0)
depth1 = torch.tensor(depth1).unsqueeze(0)
# depth0 = np.expand_dims(depth0, axis=0)
# depth1 = np.expand_dims(depth1, axis=0)
# load model
config ={}
superpoint = SuperPointNet_gauss2(config)

# predict for img0
img0 = frame2tensor(img_frame0)
pred0 = superpoint({ 'image': img0})
kps0 = pred0['keypoints'][0].unsqueeze(0)
desc0 = pred0['descriptors'][0]
scores0 = pred0['scores'][0]

print(kps0)
# warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1)
_, kpts0to1 = warp_kpts(kps0, depth0, depth1, relative_pose, intrs, intrs)

print(kpts0to1.round())
# predict for img1
img1 = frame2tensor(img_frame1)
desc1, scores1 = superpoint.computeDescriptorsAndScores({ 'image': img1, 'keypoints': kpts0to1 })

# show
# cv2.imshow('img1',img_frame1)
# cv2.imshow('img2',img_frame2)
# cv2.imshow('img_gen',img_frame1to2)
# cv2.waitKey(0)
