# coding=utf-8
# Author: nihao(nihao@bytedance.com)
# Date: 2019.08.11
#
"""Data augumentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def apply_time_warp(data, W):
  pass

def apply_fre_mask(data, F=27, m_F=2, feat_mean=None):
  data = np.copy(data)
  DIM_fre = data.shape[1]
  for i in range(m_F):
    f = np.random.randint(F)
    f0 = np.random.randint(DIM_fre - f)
    if feat_mean:
      data[:,f0: f0+f] = feat_mean[f0: f0+f]
    else:
      data[:,f0: f0+f] = 0
  return data

def apply_time_mask(data, T=100, p=0.2, m_T=2, feat_mean=None):
  data = np.copy(data)
  tao = len(data)
  for i in range(m_T):
    t = np.random.randint(T)
    t = min(t, int(tao*p))
    t0 = np.random.randint(tao - t)
    if feat_mean:
      data[t0: t0+t, : ] = feat_mean
    else:
      data[t0: t0+t, : ] = 0
  return data

