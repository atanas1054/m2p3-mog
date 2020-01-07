import numpy as np
from itertools import combinations
import math


def check_collisions(paths, frame_idx):

    curr = frame_idx[0][0]
    peds_in_frame = 1
    for f in range(1, frame_idx.shape[0]-1):

        if (curr == frame_idx[f][0]):
            peds_in_frame += 1
        else:
            L = range(f-peds_in_frame, f)
            pairs = [",".join(map(str, comb)) for comb in combinations(L, 2)]
            for pair in pairs:
                for frame in range(paths.shape[1]):
                    dist_between_peds = math.sqrt((paths[int(pair.split(",")[0])][frame][0] - paths[int(pair.split(",")[1])][frame][0]) ** 2 +
                                    (paths[int(pair.split(",")[0])][frame][1] - paths[int(pair.split(",")[1])][frame][1]) ** 2)
                    if(dist_between_peds < 0.2):
                        print(dist_between_peds)

            peds_in_frame = 1
        curr = frame_idx[f][0]
        #print(frame_idx[f])

data_train = dict(np.load("data_univ_train.npz"))
y_tr = data_train['obs_traj_rel']

x_tr = data_train['pred_traj_rel']
x_tr_ = data_train['pred_traj']
y_tr_ = data_train['obs_traj']

data_val = dict(np.load("data_univ_val.npz"))
x_val = data_val['pred_traj_rel']
x_val_ = data_val['pred_traj']
y_val = data_val['obs_traj_rel']
y_val_ = data_val['obs_traj']

data_test = dict(np.load("data_univ_test.npz"))
y_te = data_test['obs_traj_rel']
y_te_ = data_test['obs_traj']
x_te = data_test['pred_traj']
obs_vid = data_test['obs_vid']
obs_frame_idx = data_train['obs_frameidx']
seq_start_end = data_test['seq_start_end']

check_collisions(y_tr_, obs_frame_idx)


