import cv2
import torch

def read_syntheticColon_depth(path):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depth_norm = (image) / (65535 - 0)
    depth = depth_norm * 20 #0-2000mm
    depth = torch.from_numpy(depth).float()
    return depth