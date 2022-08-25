import os
import numpy as np

file_direc = 'E:\\2021-2022 Msc\Dissertation\Result\Loftr\LoFTR\matches_LOFTR'
total_match_num = np.zeros(1200)
in_match_num = np.zeros(1200)
match_ratio = np.zeros(1200)
for i in range(1,1201):
    total_match_num[i-1] = len(open(file_direc + '\\all_frame_'+ str(i).zfill(4)+'.txt', 'r').readlines())
    in_match_num[i-1] = len(open(file_direc + '\\inliers_frame_'+ str(i).zfill(4)+'.txt', 'r').readlines())
match_ratio = in_match_num / total_match_num

print('max', max(match_ratio),'in the frame', np.argmax(match_ratio))
print('min', min(match_ratio),'in the frame', np.argmin(match_ratio))
print('avg_ratio', np.mean(match_ratio))
print('avg_in', np.mean(in_match_num))
print('avg_total', np.mean(total_match_num))