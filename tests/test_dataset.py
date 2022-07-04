import sys
sys.path.append('./')

# Datasets
from datasets.sift_dataset import SIFTDataset
from datasets.superpoint_dataset import SuperPointDataset

import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test SparseDataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--train_path', type=str, default='/data/Data/COCO2017/images/train2017/',  # MSCOCO2014_yingxin
        help='Path to the directory of training imgs.')

    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')

    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='batch_size')

    opt = parser.parse_args()
    print(opt)

    train_set = SIFTDataset(opt.train_path, nfeatures=opt.max_keypoints)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

    for i, pred in enumerate(train_loader):
        print('pred = {}'.format(pred))
        # print('pred[0].shape = {}'.format(pred))

        # for k in pred:
        #     # print('k = {}'.format(k))
        #     if k != 'file_name' and k != 'image0' and k != 'image1':
        #         if type(pred[k]) == torch.Tensor:
        #             pred[k] = torch.Tensor(pred[k].cuda())
        #         else:
        #             pred[k] = torch.Tensor(torch.stack(pred[k]).cuda())

        break
