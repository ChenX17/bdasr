import os
import sys
import numpy as np
#pip install kaldi_io
import kaldi_io

cmvn_file=sys.argv[1]
feat_ark=sys.argv[2]
output_ark=sys.argv[3]
seed=int(sys.argv[4])
postfix=sys.argv[5]
np.random.seed(seed)

W=80
F=8
T=100
p=0.2
DIM=28
DIM_fre=DIM-2 #remove last 2-dim pitch

def apply_time_warp(data, W):
    pass

def apply_fre_mask(data, F, m_F, feat_mean):
    data = np.copy(data)
    for i in range(m_F):    
        f = np.random.randint(F)
        f0 = np.random.randint(DIM_fre - f)
        data[:,f0: f0+f] = feat_mean[f0: f0+f]
    return data

def apply_time_mask(data, T, p, m_T, feat_mean):
    data = np.copy(data)
    tao = len(data)
    for i in range(m_T):
        t = np.random.randint(T)
        t = min(t, int(tao*p))
        t0 = np.random.randint(tao - t)
        data[t0: t0+t, : DIM_fre] = feat_mean[: DIM_fre]
    return data

def sparse_cmvn(cmvn_file):
    with open(cmvn_file, 'r') as ff:
        lines = ff.readlines()
        assert len(lines) == 3
        mean = lines[1].strip().split()[:DIM]
        mean = np.array([float(i) for i in mean])
    return mean


mean_feat = sparse_cmvn(cmvn_file)
feature = kaldi_io.read_mat_ark(feat_ark)
with open(output_ark, 'w') as ff:
    for (fileid, feat) in feature:
        fileid = fileid+postfix
        feat = apply_fre_mask(feat, F, 2, mean_feat)
        feat = apply_time_mask(feat, T, p, 2, mean_feat)
        kaldi_io.write_mat(ff, feat, key=fileid)

