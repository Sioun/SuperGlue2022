
from pathlib import Path
import argparse

import sys
import numpy as np

import torch

from torch.autograd import Variable

# Datasets
from sift_dataset import SIFTDataset
from superpoint_dataset import SuperPointDataset

import os
import torch.multiprocessing

# torch.backends.cudnn.benchmark = True


from models.superglue import SuperGlue

import time
torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')


parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
            ' (requires ground truth pose and intrinsics)')

parser.add_argument(
    '--detector', choices={'superpoint', 'sift'}, default='superpoint',
    help='Keypoint detector')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
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
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')

parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs for evaluation')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='dump_match_pairs_coco/',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--learning_rate', type=int, default=0.00002,
    help='Learning rate')

    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='C:/datasets/train2014/', 
    help='Path to the directory of training imgs.')
# parser.add_argument(
#     '--nfeatures', type=int, default=1024,
#     help='Number of feature points to be extracted initially, in each img.')
parser.add_argument(
    '--epoch', type=int, default=1,
    help='Number of epoches')



if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'



    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
        'directory \"{}\"'.format(eval_output_dir))

    # detector_factory = {
    #     'superpoint': SuperPointDataset,
    #     'sift': SIFTDataset,
    # }
    detector_dims = {
        'superpoint': 256,
        'sift': 128,
    }

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'supgerglue': {
            # 'load_stat': 'True',
            # 'weights': 'superglue_reproduced_+5999',
            'training': 'True',
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'keypoint_encoder': [32, 64, 128, 256],
            'match_threshold': opt.match_threshold,
            'descriptor_dim': detector_dims[opt.detector],
            'GNN_layers': ['self', 'cross'] * 9
        },
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  

    superglue = SuperGlue(config.get('supgerglue', {})) 
 
    if torch.cuda.is_available():

        superglue.cuda()
        
    else:
        print("### CUDA not available ###")

    time_list = []
       
    for i in range(5):     
        pred = {'image0' : torch.rand(size=(1,1,640,480)),
        'image1': torch.rand(size=(1,1,640,480)),
        'keypoints0':torch.rand(size=(1,1,482,2)),
        'keypoints1': torch.rand(size=(1,1,482,2)),
        'descriptors0': torch.rand(256,1,482),
        'descriptors1': torch.rand(256,1,482),
        'scores0' : torch.rand(482,1),
        'scores1' : torch.rand(482,1),
        'all_matches' : torch.rand(2,1,482)
            }
        for k in pred:
            if k != 'file_name' and k!='image0' and k!='image1':
                if type(pred[k]) == torch.Tensor:
                    pred[k] = Variable(pred[k].cuda())
                else:
                    pred[k] = Variable(torch.stack(pred[k]).cuda())
          
        start_time = time.time()
        data = superglue(pred)
        end_time = time.time()
        time_list.append(end_time-start_time)
      
    print(np.mean(time_list[1:])  )