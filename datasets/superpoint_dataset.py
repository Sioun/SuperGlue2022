import sys

sys.path.append('./')

import numpy as np
import torch
import os
import cv2
import math
import datetime

# from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

# from skimage import io, transform
# from skimage.color import rgb2gray

from models.superpoint import SuperPoint
from models.utils import frame2tensor, array2tensor

class SuperPointDataset(Dataset):

    def __init__(self, image_path, image_list=None, device='cpu', superpoint_config={}):

        print('Using SuperPoint dataset')

        self.DEBUG = False
        self.image_path = image_path
        self.device = device

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(image_path)
                if name.endswith('jpg') or name.endswith('png') ]

        # Load SuperPoint model
        self.superpoint = SuperPoint(superpoint_config)
        self.superpoint.to(device)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Read image
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        height, width = image.shape[:2]
        min_size = min(height, width)

        # Transform image
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-min_size / 4, min_size / 4, size=(4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(corners, corners + warp)
        image_warped = cv2.warpPerspective(image, M, (width, height))
        if self.DEBUG: print(f'Image size: {image.shape} -> {image_warped.shape}')

        # Extract keypoints
        data = frame2tensor(image, self.device)
        pred0 = self.superpoint({ 'image': data })
        kps0 = pred0['keypoints'][0]
        desc0 = pred0['descriptors'][0]
        scores0 = pred0['scores'][0]
        if self.DEBUG: print(f'Original keypoints: {kps0.shape}, descriptors: {desc0.shape}, scores: {scores0.shape}')

        # Transform keypoints
        kps1 = cv2.perspectiveTransform(kps0.cpu().numpy()[None], M)

        # Filter keypoints
        matches = [ [], [] ]
        kps1_filtered = []
        border = self.superpoint.config.get('remove_borders', 4)
        for i, k in enumerate(kps1.squeeze()):
            if k[0] < border or k[0] >= width - border -10: continue  #wired bug
            if k[1] < border or k[1] >= height - border -10: continue
            kps1_filtered.append(k)
            matches[0].append(i)
            matches[1].append(len(matches[1]))
        all_matches = [ torch.tensor(ms) for ms in matches ]
        kps1_filtered = array2tensor(np.array(kps1_filtered), self.device)

        # Compute descriptors & scores
        data_warped = frame2tensor(image_warped, self.device)
        desc1, scores1 = self.superpoint.computeDescriptorsAndScores({ 'image': data_warped, 'keypoints': kps1_filtered })
        if self.DEBUG: print(f'Transformed keypoints: {kps1_filtered.shape}, descriptor: {desc1[0].shape}, scores: {scores1[0].shape}')

        # Draw keypoints and matches
        if self.DEBUG:
            kps0cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps0.cpu().numpy().squeeze() ]
            kps1cv = [ cv2.KeyPoint(k[0], k[1], 8) for k in kps1_filtered.cpu().numpy().squeeze() ]
            matchescv = [ cv2.DMatch(k0, k1, 0) for k0,k1 in zip(matches[0], matches[1]) ]
            outimg = None
            outimg = cv2.drawMatches(image, kps0cv, image_warped, kps1cv, matchescv, outimg)
            cv2.imwrite('matches.jpg', outimg)
            outimg = cv2.drawKeypoints(image, kps0cv, outimg)
            cv2.imwrite('keypoints0.jpg', outimg)
            outimg = cv2.drawKeypoints(image_warped, kps1cv, outimg)
            cv2.imwrite('keypoints1.jpg', outimg)

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

