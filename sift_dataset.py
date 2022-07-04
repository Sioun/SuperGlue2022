import sys

sys.path.append('./')

import numpy as np
import torch
import os
import cv2
import math
import datetime

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset

# from skimage import io, transform
# from skimage.color import rgb2gray
#
# from models.superpoint import SuperPoint
# from models.utils import frame2tensor, array2tensor


class SIFTDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, image_path, image_list = None, nfeatures = 1024):

        print('Using SIFT dataset')

        self.image_path = image_path

        # Get image names
        if image_list != None:
            with open(image_list) as f:
                self.image_names = f.read().splitlines()
        else:
            self.image_names = [ name for name in os.listdir(image_path)
                if name.endswith('jpg') or name.endswith('png') ]

        self.nfeatures = nfeatures
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=self.nfeatures)
        # self.sift = cv.xfeatures2d.SIFT_create()
        # self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)

        # self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=True)
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L1, crossCheck=False)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        # load precalculated correspondences
        # data = np.load(self.files[idx], allow_pickle=True)

        # Read image
        image = cv2.imread(os.path.join(self.image_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)

        # 使用 IO 读取图像 rgb
        # rgb_img = io.imread(file_name)
        # image = rgb2gray(rgb_img)
        sift = self.sift
        width, height = image.shape[:2]
        # max_size = max(width, height)
        corners = np.array([[0, 0], [0, height], [width, 0], [width, height]], dtype=np.float32)
        warp = np.random.randint(-224, 224, size=(4, 2)).astype(np.float32)

        M = cv2.getPerspectiveTransform(corners, corners + warp)
        warped = cv2.warpPerspective(src=image, M=M, dsize=(image.shape[1], image.shape[0]))  # return an image type

        kp1, descs1 = sift.detectAndCompute(image, None)
        kp2, descs2 = sift.detectAndCompute(warped, None)

        kp1_num = min(self.nfeatures, len(kp1))
        kp2_num = min(self.nfeatures, len(kp2))
        kp1 = kp1[:kp1_num]
        kp2 = kp2[:kp2_num]

        kp1_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp1]).astype(
            np.float32)  # maybe coordinates pt has 3 dimentions; kp1_np.shape=(50,)
        kp2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kp2]).astype(np.float32)

        if len(kp1) < 1 or len(kp2) < 1:
            # print("no kp: ",file_name)
            return {
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.float32),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.float32),
                'descriptors0': torch.zeros([0, 2], dtype=torch.float32),
                'descriptors1': torch.zeros([0, 2], dtype=torch.float32),
                'image0': image,
                'image1': warped,
                'file_name': file_name
            }
            #     descs1 = np.zeros((1, sift.descriptorSize()), np.float32)
        # if len(kp2) < 1:
        #     descs2 = np.zeros((1, sift.descriptorSize()), np.float32)

        scores1_np = np.array([kp.response for kp in kp1], dtype=np.float32)  # confidence of each key point
        scores2_np = np.array([kp.response for kp in kp2], dtype=np.float32)

        kp1_np = kp1_np[:kp1_num, :]
        kp2_np = kp2_np[:kp2_num, :]
        descs1 = descs1[:kp1_num, :]
        descs2 = descs2[:kp2_num, :]

        matched = self.matcher.match(descs1, descs2)

        kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((1, -1, 2)), M)[0, :, :]  # why [0, :, :]
        # kp1_projected = cv2.perspectiveTransform(kp1_np.reshape((-1, 2)), M) # why [0, :, :]

        dists = cdist(kp1_projected, kp2_np)

        # for mm in matched:
        #     dd = dists[mm.queryIdx, mm.trainIdx]
        #     print(dd)

        min1 = np.argmin(dists, axis=0)
        min2 = np.argmin(dists, axis=1)

        min1v = np.min(dists, axis=1)
        min1f = min2[min1v < 3]

        xx = np.where(min2[min1] == np.arange(min1.shape[0]))[0]
        matches = np.intersect1d(min1f, xx)

        missing1 = np.setdiff1d(np.arange(kp1_np.shape[0]), min1[matches])
        missing2 = np.setdiff1d(np.arange(kp2_np.shape[0]), matches)

        visualize = False
        if visualize:
            matches_dmatch = []
            for idx in range(matches.shape[0]):
                dmatch = cv2.DMatch(matches[idx], min2[matches[idx]], 0.0)
                print("Match {matches[idx]} {min2[matches[idx]]} dist={dists[matches[idx], min2[matches[idx]]]}")
                matches_dmatch.append(dmatch)
            out = cv2.drawMatches(image, kp1, warped, kp2, matches_dmatch, None)
            cv2.imshow('a', out)
            cv2.waitKey(0)

        MN = np.concatenate([min1[matches][np.newaxis, :], matches[np.newaxis, :]])
        MN2 = np.concatenate([missing1[np.newaxis, :], (len(kp2)) * np.ones((1, len(missing1)), dtype=np.int64)])
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)
        '''
        for idx in range(all_matches.shape[1]):
            pt1 = all_matches[0, idx]
            pt2 = all_matches[1, idx]
            if pt1 != self.nfeatures and pt2 != self.nfeatures:
                print(f"match: {dists[pt1, pt2]} | {pt2} {np.argmin(dists[pt1, :])} | {pt1} {np.argmin(dists[:, pt2])}")
            else:
                print(f"no match {pt1} {pt2}")
        '''
        # if kp1_np.shape != kp2_np.shape:
        #     print(kp1_np.shape, kp2_np.shape)
        #     print("MN", MN)
        #     print("MN2", MN2)
        #     print("MN3", MN3)
        #     print(" ")

        # return {'kp1': kp1_np / max_size, 'kp2': kp2_np / max_size, 'descs1': descs1 / 256., 'descs2': descs2 / 256., 'matches': all_matches}
        kp1_np = kp1_np.reshape((1, -1, 2))
        kp2_np = kp2_np.reshape((1, -1, 2))
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)

        # 归一化+通道数扩充一维
        image = torch.from_numpy(image / 255.).float().unsqueeze(0).cuda()
        warped = torch.from_numpy(warped / 255.).float().unsqueeze(0).cuda()

        return {
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'image0': image,
            'image1': warped,
            'all_matches': list(all_matches),
            'file_name': self.image_names[idx],
        }

