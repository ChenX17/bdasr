# coding=utf-8
from __future__ import print_function

import codecs
import logging
import os
import re
import glob
import random
import threading
import time
from tempfile import mkstemp

from sys import version_info
if version_info.major == 2:
  import Queue as queue
else:
  import queue as queue

import numpy as np
import tensorflow as tf
import tensorflow.contrib.framework as tff
import kaldi_io

class AttrDict(dict):
  """
  Dictionary whose keys can be accessed as attributes.
  """

  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)

  def __getattr__(self, item):
    if item not in self:
      return None
    if type(self[item]) is dict:
      self[item] = AttrDict(self[item])
    return self[item]

def expand_feed_dict(feed_dict):
  """If the key is a tuple of placeholders,
  split the input data then feed them into these placeholders.
  """
  new_feed_dict = {}
  for k, v in feed_dict.items():
    if type(k) is not tuple:
      new_feed_dict[k] = v
    else:
      # Split v along the first dimension.
      n = len(k)
      batch_size = v.shape[0]
      span = batch_size // n
      remainder = batch_size % n
      # assert span > 0
      base = 0
      for i, p in enumerate(k):
        if i < remainder:
          end = base + span + 1
        else:
          end = base + span
        new_feed_dict[p] = v[base: end]
        base = end
  return new_feed_dict


def available_variables(checkpoint_dir,from_scratch):
  all_vars = tf.global_variables()
  all_available_vars = tff.list_variables(checkpoint_dir=checkpoint_dir)
  all_available_vars = dict(all_available_vars)
  available_vars = []
  for v in all_vars:
    vname = v.name.split(':')[0]
    if from_scratch and vname == 'global_step':
      continue
    if vname in all_available_vars and v.get_shape() == all_available_vars[vname]:
      available_vars.append(v)
  return available_vars


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)
    else:
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
  return average_grads

def learning_rate_decay(config, global_step):
  """Inverse-decay learning rate until warmup_steps, then decay."""
  warmup_steps = tf.to_float(config.train.warmup_steps)
  global_step = tf.to_float(global_step)
  return config.hidden_units ** -0.5 * tf.minimum(
    (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)

def exponential_learning_rate_decay(config, global_step):
  global_step = tf.to_float(global_step)
  return config.train.learning_rate*config.train.decay_rate**(global_step/config.train.decay_step)

def platform_learning_rate_decay(config, global_step):
  warmup_steps = tf.to_float(config.train.warmup_steps)
  uniform_steps = tf.to_float(config.train.uniform_steps)
  global_step = tf.to_float(global_step)
  lr = tf.minimum((global_step + 1.0) * warmup_steps ** -1.5, warmup_steps**-0.5)
  return config.hidden_units ** -0.5 * tf.minimum(lr,(global_step-uniform_steps+1.0) ** -0.5)


def shift_right(input, pad=2):
  """Shift input tensor right to create decoder input. '2' denotes <S>"""
  input_shape = list(input.get_shape())
  return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)

def shift(input, l2r_pad=2, r2l_pad=4):
  input_shape = list(input.get_shape())
  l2r = tf.concat((tf.ones_like(input[:1, :, :1]) * l2r_pad, input[:1, :, :-1]), 2)
  r2l = tf.concat((tf.ones_like(input[1:, :, :1]) * r2l_pad, input[1:, :, :-1]), 2)
  return tf.concat([l2r,r2l], 0)
