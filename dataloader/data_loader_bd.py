# coding=utf-8
"""Data loader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import glob
import logging
import os
import re
import random
import threading
import time

import kaldi_io
import numpy as np
import six.moves.queue as queue
import tensorflow as tf
import tensorflow.contrib.framework as tff

from dataloader import data_augmentation

random.seed(8)
np.random.seed(8)

PAD_INDEX = 0
UNK_INDEX = 1
L2R_INDEX = 2
EOS_INDEX = 3
R2R_INDEX = 4

PAD = u'<PAD>'
UNK = u'<UNK>'
L2R = u'<L2R>'
EOS = u'</S>'
R2L = u'<R2L>'
MASK = u'<MASK>'


class DataLoader(object):
  """Read data and create batches for training and testing."""

  def __init__(self, config):
    # Variable init.
    self._put_done = False
    self._batch_queue = queue.Queue(100)
    self.bucket_select_dict = {}
    self.batch_bucket_limit = []
    self.uttid_target_map = {}

    # Config init.
    self._config = config
    self.apply_sentence_cmvn = self._config.apply_sentence_cmvn
    self.global_cmvn_file = self._config.global_cmvn_file
    self.global_cmvn = self._maybe_load_mean_stddev(self.global_cmvn_file)

    self.feat_files = glob.glob(self._config.train.feat_file_pattern)
    self.scp_files = glob.glob(self._config.train.feat_file_pattern)
    self.label_file = self._config.train.label_file
    self.sudo_l2r_target = self._config.train.sudo_l2r_target
    self.sudo_r2l_target = self._config.train.sudo_r2l_target

    self.frame_bucket_limit = self._config.train.frame_bucket_limit
    self.frame_bucket_limit = self.frame_bucket_limit\
        .replace('[','')\
        .replace(']','')\
        .replace(' ','')
    self.frame_bucket_limit = [int(i) for i in 
        self.frame_bucket_limit.split(',')]
    self.batch_bucket_limit_per_gpu = \
        self._config.train.batch_bucket_limit_per_gpu
    self.batch_bucket_limit_per_gpu = self.batch_bucket_limit_per_gpu\
        .replace('[','')\
        .replace(']','')\
        .replace(' ','')
    self.batch_bucket_limit_per_gpu = [int(int(i)*self._config.train.batch_factor) for i in self.batch_bucket_limit_per_gpu.split(',')]
    logging.info('frame_bucket_limit: ' + str(self.frame_bucket_limit))
    logging.info('batch_bucket_limit_per_gpu: ' + str(self.batch_bucket_limit_per_gpu))

    # Function init.
    self.load_vocab_init()
    self.select_bucket_init()
    self.load_label_init()

  def reset(self):
    """Reset Dataloader for new epoch."""
    logging.info("Reset dataloader.")
    self._put_done = False
    threading.Thread(target=self.get_training_batches_with_buckets_using_scp).start()

  def load_vocab_init(self):
    """Load vocab from disk.
    The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
    """
    def load_vocab_(path, vocab_size):
      vocab = [line.split('\t')[0] for line in codecs.open(path, 'r', 'utf-8')]
      vocab = vocab[:vocab_size]
      assert len(vocab) == vocab_size
      word2idx = {word: idx for idx, word in enumerate(vocab)}
      idx2word = {idx: word for idx, word in enumerate(vocab)}
      return word2idx, idx2word

    logging.info('Load vocabularies %s.' % (self._config.dst_vocab))
    self.dst2idx, self.idx2dst = load_vocab_(self._config.dst_vocab, self._config.dst_vocab_size)

  def _maybe_load_mean_stddev(self, mean_stddev_file):
    """Load mean, stddev from file.
    If file exist, return (mean, stddev), else return(None, None)
    """
    m = None
    s = None
    if mean_stddev_file is not None and os.path.exists(mean_stddev_file):
      with open(mean_stddev_file) as f:
        m = [float(x) for x in filter(None, f.readline().strip().split(' '))]
        s = [float(x) for x in filter(None, f.readline().strip().split(' '))]
    return (m, s)

  def select_bucket_init(self):
    frame_bucket_limit = self.frame_bucket_limit
    batch_bucket_limit_per_gpu = self.batch_bucket_limit_per_gpu
    self.batch_bucket_limit = [i*self._config.train.num_gpus for i in 
        batch_bucket_limit_per_gpu]

    self.bucket_select_dict = {}
    for index, bucket_size in enumerate(frame_bucket_limit):
      low = 0 if index == 0 else frame_bucket_limit[index-1] + 1
      high = frame_bucket_limit[index] + 1
      self.bucket_select_dict.update(dict([[i,index] for i in 
          range(low, high)]))

  def select_bucket(self, frame_size):
    if frame_size > self.frame_bucket_limit[-1]:
      return -1
    else:
      return self.bucket_select_dict[frame_size]

  def load_label_init(self):
    self.uttid_target_map = {}
    self.sudo_l2r_target_map = {}
    self.sudo_r2l_target_map = {}
    for line in codecs.open(self.label_file, 'r', 'utf-8'):
      line = line.strip()
      if line == '' or line is None:
        continue
      splits = line.strip('\n').split('\t')
      uttid = splits[0].strip()
      target = splits[1:]
      self.uttid_target_map[uttid] = target
    logging.info('loaded dst_shuf_path=' + str(self.label_file) +
        ',size=' + str(len(self.uttid_target_map)))

    for line in codecs.open(self.sudo_l2r_target, 'r', 'utf-8'):
      line = line.strip()
      if line == '' or line is None:
        continue
      splits = line.strip('\n').split('\t')
      uttid = splits[0].strip()
      sudo_target_l2r = splits[1:]
      self.sudo_l2r_target_map[uttid] = sudo_target_l2r
    logging.info('loaded dst_shuf_path=' + str(self.sudo_l2r_target) +
        ',size=' + str(len(self.sudo_l2r_target_map)))

    for line in codecs.open(self.sudo_r2l_target, 'r', 'utf-8'):
      line = line.strip()
      if line == '' or line is None:
        continue
      splits = line.strip('\n').split('\t')
      uttid = splits[0].strip()
      sudo_target_r2l = splits[1:]
      self.sudo_r2l_target_map[uttid] = sudo_target_r2l
    logging.info('loaded dst_shuf_path=' + str(self.sudo_r2l_target) +
        ',size=' + str(len(self.sudo_r2l_target_map)))

  def get_training_batches_with_buckets(self, shuffle=False):
    """Generate batches according to bucket setting."""
    # Shuffle the training files.
    src_path = self.feat_files
    dst_path = self.label_file

    # Shuffle wav not support yet, only shuffle file order.
    if shuffle:
      random.shuffle(src_path)
    src_shuf_path = src_path
    dst_shuf_path = dst_path

    # Caches to store data.
    caches = {}
    for bucket_index in range(len(self.frame_bucket_limit)):
      # Form: [src sentences, dst sentences, num_sentences].
      caches[bucket_index] = [[], [], [], [], 0]

    for index, src_shuf_file in enumerate(src_shuf_path):
      logging.info('training feat file: '
             + os.path.basename(src_shuf_file)
             + '  %d/%d' % (index+1, len(src_shuf_path)))
      ark_reader = kaldi_io.read_mat_ark(src_shuf_file)
      while True:
        #uttid, input, looped = scp_reader.read_next_utt()
        #if looped:
        #  break
        try:
          uttid, input = ark_reader.next()
        except:
          tf.logging.warn("End of file: " + src_shuf_file)
          break

        if not uttid in self.uttid_target_map:
          logging.warn('uttid=' + str(uttid) + ',target is None')
          continue
        target = self.uttid_target_map[uttid]
        if target is None:
          logging.warn('uttid=' + str(uttid) + ',target is None')
          continue

        if not uttid in self.sudo_l2r_target_map:
          logging.warn('uttid=' + str(uttid) + ',sudo_target_l2r is None')
          continue
        sudo_target_l2r = self.sudo_l2r_target_map[uttid]
        if sudo_target_l2r is None:
          logging.warn('uttid=' + str(uttid) + ',sudo_target_l2r is None')
          continue

        if not uttid in self.sudo_r2l_target_map:
          logging.warn('uttid=' + str(uttid) + ',sudo_target_r2l is None')
          continue
        sudo_target_r2l = self.sudo_r2l_target_map[uttid]
        if sudo_target_r2l is None:
          logging.warn('uttid=' + str(uttid) + ',sudo_target_r2l is None')
          continue

        if self.apply_sentence_cmvn:
          mean = np.mean(input, axis=0)
          stddev = np.std(input, axis=0)
          input = (input - mean) / stddev
        else:
          (mean, stddev) = self.global_cmvn
          if mean and stddev:
            input = (input - mean) / stddev

        ori_input_len = len(input)
        if ori_input_len < 3:
          continue
        stack_len = ori_input_len // 3
        input = input[:stack_len*3, :]
        input = input.reshape(stack_len,-1)
        target_len = len(target)

        if target_len == 0:
          continue
        if target_len > self._config.train.target_len_limit - 1:
          logging.warn(
              'uttid=' + str(uttid) + 'TOO LONG, target_len='
              + str(target_len) + ' truncated to ' +
              str(self._config.train.target_len_limit))
          target = target[:self._config.train.target_len_limit - 1]
        if stack_len < target_len:
          logging.warn('uttid=%s label is longer than wav lenth. src len: %d, '
              'label len:%d, threw it.' %(uttid, stack_len, target_len))
          continue

        bucket_index = self.select_bucket(ori_input_len)
        if bucket_index == -1:
          logging.warn('uttid=' + str(uttid) + ', frames is too long: '
                 + str(ori_input_len))
          continue
        caches[bucket_index][0].append(input)
        caches[bucket_index][1].append(target)
        caches[bucket_index][2].append(sudo_target_l2r)
        caches[bucket_index][3].append(sudo_target_r2l)
        caches[bucket_index][4] += 1

        if caches[bucket_index][4] >= self.batch_bucket_limit[bucket_index]:
          feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket_index][0])
          target_batch, target_batch_mask = self._create_bd_target_batch(caches[bucket_index][1],
          caches[bucket_index][2], caches[bucket_index][3], self.dst2idx)
          # yield (feat_batch, feat_batch_mask, target_batch, target_batch_mask)
          #yield (feat_batch, target_batch, len(caches[bucket_index][0]))
          self._batch_queue.put((feat_batch, target_batch, len(caches[bucket_index][0])))
          caches[bucket_index] = [[], [], [], [], 0]
    self._put_done = True
    del(caches)

  def get_batch(self):
    logging.info("queue size: " + str(self._batch_queue.qsize()))
    if self._batch_queue.empty() and self._put_done is True:
      return None
    while self._batch_queue.empty() and self._put_done is False:
      time.sleep(1)
      logging.info("queue size: " + str(self._batch_queue.qsize()))
    return self._batch_queue.get()

  def get_training_batches_with_buckets_using_scp(self, shuffle=False):
    """Generate batches according to bucket setting."""
    # Shuffle the training files.
    total_scp = []
    for index,scp_path in enumerate(self.scp_files):
      f = open(scp_path, 'r')
      scp = f.readlines()
      f.close()

      if index == 0:
        total_scp = scp
      else:
        total_scp += scp
    if shuffle:
      random.shuffle(total_scp)
    total_shuf_scp = total_scp

    #re-write the shuffled scp to src_shuf_path
    scp_shuf_path = self.scp_files[0].split('.')[0]+'_bd_shuf'+ str(random.randint(0,99)) +'.scp'
    f = open(scp_shuf_path,'w')
    f.writelines(total_shuf_scp)
    f.close()
    logging.info("shuffled bd double scp is stored in "+ scp_shuf_path)

    # Caches to store data.
    caches = {}
    for bucket_index in range(len(self.frame_bucket_limit)):
      # Form: [src sentences, dst sentences, num_sentences].
      caches[bucket_index] = [[], [], 0]
    
    ark_reader = kaldi_io.read_mat_scp(scp_shuf_path)
    while True:
      try:
        uttid, input = ark_reader.next()
      except:
        tf.logging.warn("End of file: " + scp_shuf_path)
        break

      #uttid_lr = uttid
      #uttid = uttid.split('_')[0]
      if not uttid in self.uttid_target_map:
        logging.warn('uttid=' + str(uttid) + ',target is None')
        continue
      target = self.uttid_target_map[uttid]
      if target is None:
        logging.warn('uttid=' + str(uttid) + ',target is None')
        continue

      if self.apply_sentence_cmvn:
        mean = np.mean(input, axis=0)
        stddev = np.std(input, axis=0)
        input = (input - mean) / stddev
      else:
        (mean, stddev) = self.global_cmvn
        if mean and stddev:
          input = (input - mean) / stddev

      ori_input_len = len(input)
      if ori_input_len < 3:
        continue
      stack_len = ori_input_len // 3
      input = input[:stack_len*3, :]
      input = input.reshape(stack_len,-1)
      target_len = len(target)

      if self._config.spec_aug is not None:
        #logging.info('use spec aug')
        if uttid.split('-')[0]=='0.9' or uttid.split('-')[0]=='1.1':
          continue
        #input = data_augmentation.apply_time_warp(input, W=20)
        #print('input is: ', input)
        input = data_augmentation.apply_fre_mask(input, F=27, m_F=2)
        input = data_augmentation.apply_time_mask(input, T=100, p=0.2, m_T=2)

      if target_len == 0:
        continue
      if target_len > self._config.train.target_len_limit - 1:
        logging.warn(
            'uttid=' + str(uttid) + 'TOO LONG, target_len='
            + str(target_len) + ' truncated to ' +
            str(self._config.train.target_len_limit))
        target = target[:self._config.train.target_len_limit - 1]
      if stack_len < target_len:
        logging.warn('uttid=%s label is longer than wav lenth. src len: %d, '
            'label len:%d, threw it.' %(uttid, stack_len, target_len))
        continue

      bucket_index = self.select_bucket(ori_input_len)
      if bucket_index == -1:
        logging.warn('uttid=' + str(uttid) + ', frames is too long: '
                + str(ori_input_len))
        continue
      caches[bucket_index][0].append(input)
      caches[bucket_index][1].append(target)
      caches[bucket_index][2] += 1

      if caches[bucket_index][2] >= self.batch_bucket_limit[bucket_index]:
        feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket_index][0])
        target_batch = self._create_bd_target_batch(caches[bucket_index][1], self.dst2idx)
        # yield (feat_batch, feat_batch_mask, target_batch, target_batch_mask)
        #yield (feat_batch, target_batch, len(caches[bucket_index][0]))
        self._batch_queue.put((feat_batch, target_batch, len(caches[bucket_index][0])))
        caches[bucket_index] = [[], [], 0]
    os.remove(scp_shuf_path)
    self._put_done = True
    del(caches)

  def _create_feat_batch(self, indices):
    # Pad to the same length.
    # indices的数据是[[feat_len1, feat_dim], [feat_len2, feat_dim], ...]
    assert len(indices) > 0
    batch_size = len(indices)
    maxlen = max([len(s) for s in indices])
    feat_dim = indices[0].shape[1]
    feat_batch = np.zeros([batch_size, maxlen, feat_dim], dtype=np.float32)
    feat_batch.fill(PAD_INDEX)
    feat_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
    for i in range(batch_size):
      feat = indices[i]
      feat_len, feat_dim = np.shape(feat)
      feat_batch[i,:feat_len, :] = np.copy(feat)
      feat_batch_mask[i, :feat_len] = 0
    return feat_batch, feat_batch_mask

  def _create_target_batch(self, sents, phone2idx):
    # sents的数据是[word1 word2 ... wordn]
    indices = []
    for sent in sents:
      x = []

      for word in (sent + [EOS]):
        x_tmp = phone2idx.get(word, UNK_INDEX)
        x.append(x_tmp)
        if x_tmp == UNK_INDEX and word != UNK:
          logging.warn('=========[ZSY]x_tmp=UNK_INDEX, word=' + str(word.encode('UTF-8')))
      indices.append(x)

    # Pad to the same length.
    batch_size = len(sents)
    maxlen = max([len(s) for s in indices])
    target_batch = np.zeros([batch_size, maxlen], np.int32)
    target_batch.fill(PAD_INDEX)
    target_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
    for i, x in enumerate(indices):
      target_batch[i, :len(x)] = x
      target_batch_mask[i, :len(x)] = 0
    return target_batch, target_batch_mask
  
  def _create_bd_target_batch(self, sents, phone2idx):
    # sents的数据是[word1 word2 ... wordn]
    indices = []
    for sent in sents:
      x = []

      for word in (sent + [EOS]):
        x_tmp = phone2idx.get(word, UNK_INDEX)
        x.append(x_tmp)
        if x_tmp == UNK_INDEX and word != UNK:
          logging.warn('=========[ZSY]x_tmp1=UNK_INDEX, word=' + str(word.encode('UTF-8')))
      indices.append(x)

    indices_with_mask_l2r = []
    indices_with_mask_r2l = []
    mask_rate = self._config.mask_rate
    mask_prob = self._config.mask_prob
    for sent in sents:
      x = []
      y = []
      for word in (sent + [EOS]):
        x_tmp = phone2idx.get(word, UNK_INDEX)
        x.append(x_tmp)
      indices_with_mask_l2r.append(x)
      for word in (sent + [EOS]):
        y_tmp = phone2idx.get(word, UNK_INDEX)
        y.append(y_tmp)
        if y_tmp == UNK_INDEX and word != UNK:
          logging.warn('=========[ZSY]x_tmp3=UNK_INDEX, word=' + str(word.encode('UTF-8')))
      indices_with_mask_r2l.append(y)

    # Pad to the same length.
    batch_size = len(sents)
    
    maxlen = max([len(s) for s in indices])
    target_batch_l2r = np.zeros([batch_size, maxlen], np.int32)
    target_batch_r2l = np.zeros([batch_size, maxlen], np.int32)
    target_batch_l2r.fill(PAD_INDEX)
    target_batch_r2l.fill(PAD_INDEX)
    for i, x in enumerate(indices):
      target_batch_l2r[i, :len(x)] = x
      target_batch_r2l[i, :len(x)] = x[:-1][::-1]+[3]

    # masked_target_batch_l2r = np.zeros([batch_size, maxlen], np.int32)
    # masked_target_batch_r2l = np.zeros([batch_size, maxlen], np.int32)
    # masked_target_batch_l2r.fill(PAD_INDEX)
    # masked_target_batch_r2l.fill(PAD_INDEX)
    # for i, x in enumerate(indices_with_mask_l2r):
    #   masked_target_batch_l2r[i, :len(x)] = x
    #   masked_target_batch_r2l[i, :len(x)] = indices_with_mask_r2l[i][:-1][::-1]+[3]

    target_batch_l2r = target_batch_l2r[:, np.newaxis, :]
    #new_target = target_batch_l2r
    target_batch_r2l = target_batch_r2l[:, np.newaxis, :]
    new_target = np.concatenate((target_batch_l2r, target_batch_r2l), 1)

    # masked_target_batch_l2r = masked_target_batch_l2r[:, np.newaxis, :]
    # masked_target_batch_r2l = masked_target_batch_r2l[:, np.newaxis, :]
    # masked_target = np.concatenate((masked_target_batch_l2r, masked_target_batch_r2l), 1)
    # return masked_target, new_target
    return new_target


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

DataReader=DataLoader
