# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""Data augumentation."""


import numpy as np
import random
import tensorflow as tf

  
def fre_mask(data, F=27, m_F=2):
  v = data.shape[1]
  for i in range(m_F):
    f = np.random.randint(F)
    f0 = np.random.randint(v-f)
    data[:,f0: f0+f] = 0
  return data

def time_mask(data, T=100, p=0.2, m_T=2):
  tau = data.shape[0]
  for i in range(m_T):
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, tau - t)
    data[t0:t0+tau, :] = 0
  return data
