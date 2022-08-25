import sys
sys.path.append('./')

# Datasets
from datasets.sift_dataset import SIFTDataset
from datasets.superpoint_dataset import SuperPointDataset

import argparse

import torch
from models.superpoint import SuperPoint

from models.utils import read_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test SparseDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--image_path', type=str, default='assets/2799_matches.png',  # MSCOCO2014_yingxin
        help='Path to test image.')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
             ' (Must be positive)')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    opt = parser.parse_args()
    print(opt)

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        # 'superglue': {
        #     'weights': opt.superglue,
        #     'sinkhorn_iterations': opt.sinkhorn_iterations,
        #     'match_threshold': opt.match_threshold,
        # }
    }

    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    superpoint = SuperPoint(config.get('superpoint', {})).cuda()

    # Load the image pair.
    image0, inp0, scales0 = read_image(
        opt.image_path, opt.resize, 0, 1.0)

    print('inp0 = {}'.format(inp0))

    pred = {}

    # Extract SuperPoint (keypoints, scores, descriptors) if not provided
    pred0 = superpoint({'image': inp0})
    pred = {**pred, **{k + '0': v for k, v in pred0.items()}}

    print('pred = {}'.format(pred))