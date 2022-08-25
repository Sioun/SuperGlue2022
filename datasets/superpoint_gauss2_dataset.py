import sys

sys.path.append('./')

import numpy as np
import torch
import os
import cv2
import math
import datetime
from pathlib import Path
from utils.read_depth import *
from utils.read_poses import *
from utils.geometry import *

# from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

# from skimage import io, transform
# from skimage.color import rgb2gray

from models.SuperPointNet_gauss2 import SuperPointNet_gauss2
from models.utils import frame2tensor, array2tensor

class SuperPoint_Guass2_Dataset(Dataset):

    def __init__(self, image_path, image_list=None, device='cpu', superpoint_config={}, train=True):

        print('Using SuperPoint_gauss2 dataset')

        self.DEBUG = False
        self.image_path = Path(image_path)
        self.device = device
        self.train = train
        self.homo = False

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            print(image_path)
            scene_list_path = (
                self.image_path / "train.txt"
                if self.train == True
                else self.image_path / "val.txt"
            )
            self.scenes = [
                # (label folder, raw image path)
                (Path(self.image_path / folder[:-1]), Path(self.image_path / folder[:-1])) \
                for folder in open(scene_list_path)
            ]
            self.image_names = list()
            self.depth_names = list()
            for (scene, scene_img_folder) in self.scenes:
                for image_pathi in scene_img_folder.iterdir():
                    if 'FrameBuffer' in str(image_pathi):
                        self.image_names.append(str(image_pathi))
                    if 'Depth' in str(image_pathi):
                        self.depth_names.append(str(image_pathi))
            self.intrs = dict()
            with open(image_path + '\camS.txt', 'r') as f:
                l = [[float(num) for num in line.split(' ')] for line in f]
                self.intrs['s'] = l
            with open(image_path + '\camB.txt', 'r') as f:
                l = [[float(num) for num in line.split(' ')] for line in f]
                self.intrs['b'] = l

        # Load SuperPoint model
        self.superpoint = SuperPointNet_gauss2(superpoint_config)
        self.superpoint.to(device)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Read image
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        min_size = min(height, width)

        # Get the warped image or Get the next-frame image
        # if use homography
        if self.homo:
            # Transform image
            corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
            warp = np.random.randint(-min_size / 2, min_size / 2, size=(4, 2)).astype(np.float32)
            M = cv2.getPerspectiveTransform(corners, corners + warp)
            image_warped = cv2.warpPerspective(image, M, (width, height))
            if self.DEBUG: print(f'Image size: {image.shape} -> {image_warped.shape}')
        #if use image pairs
        else:
            scene = self.image_names[idx][-24:-21].replace('_', '')
            cur_frame_num = int(self.image_names[idx][-8:-4])
            if cur_frame_num >= 1200:
                return 0
            # if cur_frame_num % 10 == 1:
            #     return 0
            next_frame = str(cur_frame_num + 1).zfill(4)
            next_frame_name = self.image_names[idx][:-8]+ next_frame + '.png'
            image_warped = cv2.imread(os.path.join(self.image_path, next_frame_name), cv2.IMREAD_GRAYSCALE)
            if self.DEBUG:
                print('current scene:',scene)
                print('current name:',self.image_names[idx])
                print('next name:', next_frame_name)



        # Extract keypoints
        data = frame2tensor(image, self.device)
        pred0 = self.superpoint({'image': data})
        kps0 = pred0['keypoints'][0]
        desc0 = pred0['descriptors'][0]
        scores0 = pred0['scores'][0]
        if self.DEBUG:
            print(f'Original keypoints: {kps0.shape}, descriptors: {desc0.shape}, scores: {scores0.shape}')
            print('test for kp:', kps0)
            print('test for score:', scores0)

        # if use homography, then Transform keypoints
        if self.homo:
            kps1 = cv2.perspectiveTransform(kps0.cpu().numpy()[None], M)
        # if use next-frame image, calculate keypoints from depth and camera intrs
        else:
            # TODO: make them public
            poses_quat, poses_mat = get_poses(scene, str(self.image_path))
            frame0 = poses_mat[cur_frame_num, :, :]
            frame1 = poses_mat[cur_frame_num+1, :, :]
            if self.DEBUG:
                print('frame1',frame1)

            # get 1x3x4 relative pos matrix
            relative_pose = get_relative_pose(frame1, frame0)
            relative_pose = torch.tensor(relative_pose[0:3, :].astype(np.float32)).unsqueeze(0)
            if self.DEBUG:
                print('relative pose',relative_pose)

            # get 1xHxW depth map
            depth0 = read_syntheticColon_depth(os.path.join(self.image_path, self.depth_names[idx]))
            next_depth_name = self.depth_names[idx][:-8] + next_frame + '.png'
            depth1 = read_syntheticColon_depth(os.path.join(self.image_path, next_depth_name))

            depth0 = torch.tensor(depth0).unsqueeze(0)
            depth1 = torch.tensor(depth1).unsqueeze(0)

            # get 1x3x3 intrinsic matrix
            if scene[0] == 'S':
                intrs = self.intrs['s']
            else:
                intrs = self.intrs['b']
            intrs = torch.from_numpy(np.resize(intrs, (3, 3)).astype(np.float32)).unsqueeze(0)
            kps0 = kps0.unsqueeze(0).cpu()
            _, kps1 = warp_kpts(kps0, depth0, depth1, relative_pose, intrs, intrs)
            kps1 = kps1.round()

        # Filter keypoints
        matches = [[], []]
        kps1_filtered = []
        border = 0
        # n = 0
        for i, k in enumerate(kps1.squeeze()):
                # n += 1
                if k[0] <= border or k[0] >= width - 4:
                    continue
                if k[1] <= border or k[1] >= height - 4:
                    continue
                if self.homo == False:
                    k = k.numpy()
                kps1_filtered.append(k)
                matches[0].append(i)
                matches[1].append(len(matches[1]))
                # if n > 100: break
        all_matches = [torch.tensor(ms) for ms in matches]
        kps1_filtered = array2tensor(np.array(kps1_filtered), self.device)
        if self.DEBUG:
            print('filterd kp:',kps1_filtered.shape)
            print(kps1_filtered)

        # Compute descriptors & scores
        data_warped = frame2tensor(image_warped, self.device)
        desc1, scores1 = self.superpoint.computeDescriptorsAndScores({ 'image': data_warped, 'keypoints': kps1_filtered })
        if self.DEBUG: print(f'Transformed keypoints: {kps1_filtered.shape}, descriptor: {desc1[0].shape}, scores: {scores1[0].shape}')

        # Draw keypoints and matches
        if self.DEBUG:
            # kps0cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze()[:110,:] ]
            # kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps1_filtered.cpu().numpy().squeeze()[:100,:]]
            # matchescv = [ cv2.DMatch(k0, k1, 0) for k0,k1 in zip(matches[0][:100], matches[1][:100]) ]
            print('score 0 in shape', scores0.shape,' is',scores0)
            print('score 1  in shape', scores1[0].shape,'is', scores1)
            kps0cv = [cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze()]
            kps1cv = [cv2.KeyPoint(k[0], k[1], 8) for k in kps1_filtered.cpu().numpy().squeeze()]
            matchescv = [cv2.DMatch(k0, k1, 0) for k0, k1 in zip(matches[0], matches[1])]
            outimg = None
            outimg = cv2.drawMatches(image, kps0cv, image_warped, kps1cv, matchescv, outimg)
            cv2.imwrite('matches.jpg', outimg)
            outimg = cv2.drawKeypoints(image, kps0cv, outimg)
            cv2.imwrite('keypoints0.jpg', outimg)
            outimg = cv2.drawKeypoints(image_warped, kps1cv, outimg)
            cv2.imwrite('keypoints1.jpg', outimg)

        if self.train == False:
            return {
                'keypoints0': kps0.unsqueeze(0),
                'keypoints1': kps1_filtered,
                'descriptors0': list(desc0),
                'descriptors1': list(desc1[0]),
                'scores0': list(scores0),
                'scores1': list(scores1[0]),
                'image0': data,
                'image1': data_warped,
                'all_matches': all_matches,
                'file_name': self.image_names[idx],
            }

        return {
            'keypoints0': kps0.unsqueeze(0),
            'keypoints1': kps1_filtered,
            'descriptors0': list(desc0),
            'descriptors1': list(desc1[0]),
            'scores0': list(scores0),
            'scores1': list(scores1[0]),
            'image0': data.squeeze(0),
            'image1': data_warped.squeeze(0),
            'all_matches': all_matches,
            'file_name': self.image_names[idx],
        }

