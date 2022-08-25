
from pathlib import Path
import argparse
import random
import sys
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
from torch.autograd import Variable

# Datasets
from sift_dataset import SIFTDataset
from superpoint_dataset import SuperPointDataset
from datasets.superpoint_gauss2_dataset import SuperPoint_Guass2_Dataset

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.multiprocessing
from tqdm import tqdm
import torch.cuda.profiler as profiler
# import pyprof
# pyprof.init()
# torch.backends.cudnn.benchmark = True

# from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.matchingForTraining import MatchingForTraining

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.multiprocessing.set_start_method("spawn")
# torch.cuda.set_device(0)

# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass


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
    '--detector', choices={'superpoint', 'sift','superpoint_gauss2'}, default='superpoint_gauss2',
    help='Keypoint detector')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.015,
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
    '--resize', type=int, nargs='+', default=[475, 475],
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
    '--eval_output_dir', type=str, default='cam/',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--learning_rate', type=int, default=0.00002,
    help='Learning rate')

    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='E:\\2021-2022 Msc\dataset\SyntheticColon_I',
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--val_path', type=str, default='E:\\2021-2022 Msc\dataset\SyntheticColon_I',
    help='Path to the directory of validation imgs.')
# parser.add_argument(
#     '--nfeatures', type=int, default=1024,
#     help='Number of feature points to be extracted initially, in each img.')
parser.add_argument(
    '--epoch', type=int, default=10,
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
        'superpoint_gauss2':256
    }

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superpoint_gauss2': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'supgerglue': {
            'load_stat': 'True',
            'weights': 'superglue_indoor',
            'training': 'True',
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'keypoint_encoder': [32, 64, 128, 256],
            'match_threshold': opt.match_threshold,
            'descriptor_dim': detector_dims[opt.detector],
            'GNN_layers': ['self', 'cross'] * 9
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data loader
    if opt.detector == 'superpoint':
        train_set = SuperPointDataset(opt.train_path, device=device, superpoint_config=config.get('superpoint', {}))
    elif opt.detector == 'sift':
        train_set = SIFTDataset(opt.train_path, nfeatures=opt.max_keypoints)
    elif opt.detector == 'superpoint_gauss2':
        train_set = SuperPoint_Guass2_Dataset(opt.train_path, device=device, superpoint_config=config.get('superpoint_gauss2', {}),train= True)
        val_set = SuperPoint_Guass2_Dataset(opt.val_path, device=device,
                                              superpoint_config=config.get('superpoint_gauss2', {}),train= False)
    else:
        RuntimeError('Error detector : {}'.format(opt.detector))

    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=opt.batch_size, drop_last=True)

    # superpoint = SuperPoint(config.get('superpoint', {}))
    superglue = SuperGlue(config.get('supgerglue', {}))
    # teacher = SuperGlue(config.get('teacher', {}))
    if torch.cuda.is_available():
        # superpoint.cuda()
        superglue.cuda()
        
    else:
        print("### CUDA not available ###")
    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)

    mean_loss = []
    print('training start')
    for epoch in range(1, opt.epoch+1):
        epoch_loss = 0
        superglue.train()
        # train_loader = tqdm(train_loader)
        for i, pred in enumerate(train_loader):
            if pred == 0:
                print('pass')
                continue
            for k in pred:
                if k != 'file_name' and k!='image0' and k!='image1':
                    if type(pred[k]) == torch.Tensor:
                        # print('yes for',k)
                        # keypoints0 keypoints1
                        pred[k] = Variable(pred[k].cuda())
                    else:
                        pred[k] = Variable(torch.stack(pred[k]).cuda())
            # print(pred['image0'].shape)
            data = superglue(pred)
            for k, v in pred.items():
                pred[k] = v[0]
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has no keypoint
                continue

            superglue.zero_grad()
            Loss = pred['loss']

            epoch_loss += Loss.item()
            mean_loss.append(Loss) # every 10 pairs
            Loss.backward()
            optimizer.step()


            # visualization
            if (i+1) % 100 == 0:
            # if True:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item()))   # Loss.item()    
                mean_loss = []

                ### eval ###
                # Visualize the matches.
                superglue.eval()
                image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()
                matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
                image0 = read_image_modified(image0, opt.resize, opt.resize_float)
                image1 = read_image_modified(image1, opt.resize, opt.resize_float)
                # ------- debug -------
                # for m in range(0, len(matches)):
                #     if m > 100:
                #         matches[m] = -1
                # ------ end -------
                valid = matches > -1
                print(valid.shape)
                mkpts0 = kpts0.squeeze()[valid]
                mkpts1 = kpts1.squeeze()[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir / '{}_{}_matches.{}'.format(str(epoch),str(i), opt.viz_extension)

                c = ["darkred", "red", "lightcoral", "white", "palegreen", "green", "darkgreen"]
                v = [0, .15, .4, .5, 0.6, .9, 1.]
                l = list(zip(v, c))
                cmap = LinearSegmentedColormap.from_list('rg', l, N=256)
                color = cmap(mconf)
                stem = pred['file_name']
                text = []

                make_matching_plot(
                    image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                    text, viz_path, stem, stem, opt.show_keypoints,
                    opt.fast_viz, opt.opencv_display, 'Matches')

            # model save
            if (i+1) % 1000 == 0:
                model_out_path = "logs/superglue_indoor_{}_{}.pth".format(epoch,i)
                torch.save(superglue.state_dict(), model_out_path)
                print ('Epoch [{}/{}], Step [{}/{}], Checkpoint saved to {}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), model_out_path)) 

        epoch_loss /= len(train_loader)
        model_out_path = "exp/model_out_epoch_{}.pth".format(epoch)
        torch.save(superglue, model_out_path)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
            .format(epoch, opt.epoch, epoch_loss, model_out_path))
        

