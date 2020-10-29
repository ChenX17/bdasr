# coding=utf-8
"""Data loader for test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import logging
import glob
import os

import kaldi_io
import numpy as np
import tensorflow as tf

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super(AttrDict, self).__init__(*args, **kwargs)
  def __getattr__(self, item):
    if item not in self:
      return None
    if type(self[item]) is dict:
      self[item] = AttrDict(self[item])
    return self[item]


class DataReader(object):
  """Read data and create batches for training and testing."""

  def __init__(self, config):
    self._config = config
    self.load_vocab()
    self.feat_files = glob.glob(self._config.train.feat_file_pattern)
    self.apply_sentence_cmvn = self._config.apply_sentence_cmvn
    self.global_cmvn_file = self._config.global_cmvn_file
    self.global_cmvn = self._maybe_load_mean_stddev(self.global_cmvn_file)
    self.label_file = self._config.test.set1.label_file
    self.test_feat_files = glob.glob(self._config.test.set1.feat_file_pattern)

  def load_vocab(self):
    """
    Load vocab from disk.
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
    """
    Load mean, stddev from file.
    If file exist, return (mean, stddev), else return(None, None)
    """

    m = None
    s = None
    if mean_stddev_file is not None and os.path.exists(mean_stddev_file):
      with open(mean_stddev_file) as f:
        m = [float(x) for x in filter(None, f.readline().strip().split(' '))]
        s = [float(x) for x in filter(None, f.readline().strip().split(' '))]
    return (m, s)


  def _create_feat_batch(self, indices):
    # Pad to the same length.
    # indices的数据是[[feat_len1, feat_dim], [feat_len2, feat_dim], ...]
    assert len(indices) > 0
    batch_size = len(indices)
    maxlen = max([len(s) for s in indices])
    feat_dim = indices[0].shape[1]
    feat_batch = np.zeros([batch_size, maxlen, feat_dim], dtype=np.float32)
    #feat_batch.fill(PAD_INDEX)
    feat_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
    for i in range(batch_size):
      feat = indices[i]
      feat_len, feat_dim = np.shape(feat)
      feat_batch[i,:feat_len, :] = np.copy(feat)
      feat_batch_mask[i, :feat_len] = 0
    return feat_batch, feat_batch_mask

  def get_test_batches(self, src_path, batch_size):
    #logging.info('-----------get_test_batches-------------')
    #src_path = self.test_feat_files
    src_path = glob.glob(src_path)
    for index, src_file in enumerate(src_path):
      logging.info('testing feat file: '
          + os.path.basename(src_file)
          + ' %d/%d' % (index+1,len(src_path)))
      ark_reader = kaldi_io.read_mat_ark(src_file)
      cache = []
      uttids = []
      while True:
        try:
          uttid, input = ark_reader.next()
        except:
          tf.logging.warn("End of file: " + src_file)
          break
        if self.apply_sentence_cmvn:
          mean = np.mean(input, axis=0)
          stddev = np.std(input, axis=0)
          input = (input - mean) / stddev
        else:
          (mean, stddev) = self.global_cmvn
          if mean and stddev:
            input = (input - mean) / stddev
        ori_input_len = len(input)
        stack_len = ori_input_len // 3
        input = input[:stack_len*3, :]
        input = input.reshape(stack_len,-1)

        cache.append(input)
        uttids.append(uttid)
        if len(cache) >= batch_size:
          feat_batch, feat_batch_mask = self._create_feat_batch(cache)
          yield feat_batch, uttids
          cache = []
          uttids = []
      if cache:
        feat_batch, feat_batch_mask = self._create_feat_batch(cache)
        yield feat_batch, uttids
  def get_test_batches_with_buckets(self, src_path, tokens_per_batch):
    src_path = self.test_feat_files
    buckets = [(i) for i in range(50, 10000, 10)]

    def select_bucket(sl):
      for l1 in buckets:
        if sl < l1:
          return l1
      raise Exception("The sequence is too long: ({})").format(sl)
    for index, src_file in enumerate(src_path):
      logging.info('testing feat file: '
          + os.path.basename(src_file)
          + ' %d/%d' % (index+1,len(src_path)))
      caches = {}
      for bucket in buckets:
        caches[bucket] = [[], [], 0]
      ark_reader = kaldi_io.read_mat_ark(src_file)
      count = 0
      while True:
        try:
          uttid, input = ark_reader.next()
        except:
          tf.logging.warn("End of file: " + src_file)
          break
        if self.apply_sentence_cmvn:
          mean = np.mean(input, axis=0)
          stddev = np.std(input, axis=0)
          input = (input - mean) / stddev
        else:
          (mean, stddev) = self.global_cmvn
          if mean and stddev:
            input = (input - mean) / stddev

        ori_input_len = len(input)
        stack_len = ori_input_len // 3
        input = input[:stack_len*3, :]
        input = input.reshape(stack_len,-1)
        bucket = select_bucket(ori_input_len)
        caches[bucket][0].append(input)
        caches[bucket][1].append(uttid)
        caches[bucket][2] += ori_input_len
        count += 1

        if caches[bucket][2] > tokens_per_batch:
          feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
          yield feat_batch, caches[bucket][1]
          caches[bucket] = [[], [], 0]

        #clean remain sentences.
      for bucket in buckets:
        if len(caches[bucket][0]) > 0:
          logging.info('get_test_batches_with_buckets, len(caches[bucket][0])=' + str(len(caches[bucket][0])))
          feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
          yield feat_batch, caches[bucket][1]

      logging.info('get_test_batches_with_buckets, loaded count=' + str(count))
  def get_test_batches_with_tokens_per_batch(self, src_path, tokens_per_batch):
    '''get test batch when the number of frames exceed tokens_per_batch'''
    src_path = glob.glob(src_path)
    for index, src_file in enumerate(src_path):
      logging.info('testing feat file: '
          + os.path.basename(src_file)
          + ' %d/%d' % (index+1,len(src_path)))
      ark_reader = kaldi_io.read_mat_ark(src_file)
      cache = []
      uttids = []
      count = 0
      while True:
        try:
          uttid, input = ark_reader.next()
        except:
          tf.logging.warn("End of file: " + src_file)
          break
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
        count += stack_len

        cache.append(input)
        uttids.append(uttid)
        if count >= tokens_per_batch:
          feat_batch, feat_batch_mask = self._create_feat_batch(cache)
          yield feat_batch, uttids
          count = 0
          cache = []
          uttids = []
      if cache:
        feat_batch, feat_batch_mask = self._create_feat_batch(cache)
        yield feat_batch, uttids
      del(cache)
  def indices_to_words(self, Y, o='dst'):
    assert o in ('src', 'dst')
    idx2word = self.idx2src if o == 'src' else self.idx2dst
    sents = []
    for y in Y:  # for each sentence
      sent = []
      for i in y:  # For each word
        if i == 3:  # </S>
          break
        w = idx2word[i]
        sent.append(w)
      sents.append(''.join(sent))
    return sents


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
