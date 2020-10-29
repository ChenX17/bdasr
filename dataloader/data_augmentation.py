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


def apply_time_warp(data, W):
  
  data = np.copy(data)
  print(data.shape)
  data = data.transpose(1,0)
  v, tau = data.shape[0], data.shape[1]

  center_position = v/2
  random_point = np.random.randint(low=W, high=tau - W)
  # warping distance chose.
  w = np.random.uniform(low=0, high=W)

  control_point_locations = [[center_position, random_point]]
  control_point_locations = np.float32(np.expand_dims(control_point_locations, 0))

  control_point_destination = [[center_position, random_point + w]]
  control_point_destination = np.float32(np.expand_dims(control_point_destination, 0))
  data = np.expand_dims(np.expand_dims(data, axis=0), axis=-1)

  g = tf.Graph()

  with g.as_default():

    mel_spectrogram_holder = tf.placeholder(tf.float32, shape=[1, v, tau, 1])
    location_holder = tf.placeholder(tf.float32, shape=[1, 1, 2])
    destination_holder = tf.placeholder(tf.float32, shape=[1, 1, 2])
    
    
    warped_mel_spectrogram_op, _ = sparse_image_warp(mel_spectrogram_holder,
                                                      source_control_point_locations=location_holder,
                                                      dest_control_point_locations=destination_holder,
                                                      interpolation_order=2,
                                                      regularization_weight=0,
                                                      num_boundary_points=1
                                                      )
  # Change warp result's data type to numpy array for masking step.
  feed_dict = {mel_spectrogram_holder:data,
                location_holder:control_point_locations,
                destination_holder:control_point_destination}
  with tf.Session(graph=g) as sess:
    warped_mel_spectrogram = sess.run(warped_mel_spectrogram_op, feed_dict=feed_dict)

  warped_mel_spectrogram = warped_mel_spectrogram.reshape([warped_mel_spectrogram.shape[1],
                                                            warped_mel_spectrogram.shape[2]])
  print(warped_mel_spectrogram.transpose(1, 0).shape)
  return warped_mel_spectrogram.transpose(1, 0)
  

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

