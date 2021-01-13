# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
Date: 2020-10-26 15:04:03
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2020-10-29 20:21:06
'''

"""Data augumentation."""


import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.image import sparse_image_warp


def time_warp(data, W=80):
  data = np.copy(data)
  print(data.shape)
  data = data.transpose(1,0)
  v, tau = data.shape[0], data.shape[1]

  data = np.reshape(data, (1, v, tau, 1))

  horiz_line_thru_ctr = data[0][v//2]

  print('W is :', W)
  print('tau-W is :', tau-W)

  random_pt = horiz_line_thru_ctr[random.randrange(W, tau - W)] # random point along the horizontal/time axis
  w = np.random.uniform(-W, W) # distance
  print('w is :', w)
    
  # Source Points
  src_points = [[[v//2, random_pt[0]]]]
    
  # Destination Points
  dest_points = [[[v//2, random_pt[0] + w]]]

  print(data.dtype)
  data = data.astype(np.float32)
  print(data.dtype)
    
  warped_mel_spectrogram, _ = sparse_image_warp(data, src_points, dest_points, num_boundary_points=2)

  # print(warped_mel_spectrogram.numpy().transpose(1, 0).shape)
  return tf.transpose(tf.squeeze(warped_mel_spectrogram), perm=[1, 0])
  # return warped_mel_spectrogram.transpose(1, 0)
  
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
