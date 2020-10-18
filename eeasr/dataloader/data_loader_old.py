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

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

PAD = u'<PAD>'
UNK = u'<UNK>'
BOS = u'<S>'
EOS = u'</S>'


class DataReader(object):
  """Read data and create batches for training and testing."""

  def __init__(self, config):
    self._config = config
    self.load_vocab()
    self.feat_files = glob.glob(self._config.train.feat_file_pattern)
    self.apply_sentence_cmvn = self._config.apply_sentence_cmvn
    self.global_cmvn_file = self._config.global_cmvn_file
    self.global_cmvn = self._maybe_load_mean_stddev(self.global_cmvn_file)
    self.label_file = self._config.train.label_file
    self.frame_bucket_limit = self._config.train.frame_bucket_limit
    self.frame_bucket_limit = self.frame_bucket_limit\
        .replace('[','')\
        .replace(']','')\
        .replace(' ','')
    self.frame_bucket_limit = [int(i) for i in self.frame_bucket_limit.split(',')]
    self.batch_bucket_limit_per_gpu = self._config.train.batch_bucket_limit_per_gpu
    self.batch_bucket_limit_per_gpu = self.batch_bucket_limit_per_gpu\
        .replace('[','')\
        .replace(']','')\
        .replace(' ','')
    self.batch_bucket_limit_per_gpu = [int(int(i)*self._config.train.batch_factor) for i in self.batch_bucket_limit_per_gpu.split(',')]
    logging.info('frame_bucket_limit: ' + str(self.frame_bucket_limit))
    logging.info('batch_bucket_limit_per_gpu: ' + str(self.batch_bucket_limit_per_gpu))
    self._batch_queue = queue.Queue(100)
    self._put_done = False
    if self._config.train.debug:
      self.batch_bucket_limit_per_gpu = [8] * len(self.batch_bucket_limit_per_gpu)

  def reset(self):
    """Reset Dataloader for new epoch."""
    logging.info("Reset dataloader.")
    self._put_done = False
    threading.Thread(target=self.get_training_batches_with_buckets).start()

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

  def get_training_batches_with_buckets(self, shuffle=True):
    """
    Generate batches according to bucket setting.
    """

    # buckets = [(i, i) for i in range(5, 1000000, 3)]
    # buckets = [(i, i) for i in range(self._config.bucket_min, self._config.bucket_max, self._config.bucket_step)]

    frame_bucket_limit = self.frame_bucket_limit
    batch_bucket_limit_per_gpu = self.batch_bucket_limit_per_gpu
    batch_bucket_limit = [i*self._config.train.num_gpus for i in batch_bucket_limit_per_gpu]

    bucket_select_dict = {}
    for index, bucket_size in enumerate(frame_bucket_limit):
      low = 0 if index == 0 else frame_bucket_limit[index-1] + 1
      high = frame_bucket_limit[index] + 1
      bucket_select_dict.update(dict([[i,index] for i in range(low, high)]))

    def select_bucket(frame_size):
      if frame_size > frame_bucket_limit[-1]:
        return -1
      else:
        return bucket_select_dict[frame_size]

    # Shuffle the training files.
    src_path = self.feat_files
    dst_path = self.label_file

    # Shuffle wav not support yet, only shuffle file order
    if shuffle:
      random.shuffle(src_path)
    src_shuf_path = src_path
    dst_shuf_path = dst_path

    # Caches to store data
    caches = {}
    for bucket_index in range(len(frame_bucket_limit)):
      # Former: [src sentences, dst sentences, num_sentences].
      caches[bucket_index] = [[], [], 0]

    uttid_target_map = {}
    for line in codecs.open(dst_shuf_path, 'r', 'utf-8'):
      line = line.strip()
      if line == '' or line is None:
        continue
      #splits = re.split('\s+', line)
      splits = line.strip('\n').split('\t')
      uttid = splits[0].strip()
      target = splits[1:]
      uttid_target_map[uttid] = target
    logging.info('loaded dst_shuf_path=' + str(dst_shuf_path) +
        ',size=' + str(len(uttid_target_map)))

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

        if not uttid in uttid_target_map:
          logging.warn('uttid=' + str(uttid) + ',target is None')
          continue
        target = uttid_target_map[uttid]
        if target is None:
          logging.warn('uttid=' + str(uttid) + ',target is None')
          continue
        if self.apply_sentence_cmvn:
          mean = np.mean(input, axis=0)
          stddev = np.std(input, axis=0)
          input = (input - mean) / stddev
        else:
          continue
          (mean, stddev) = self.global_cmvn
          if mean and stddev:
            input = (input - mean) / stddev

        ori_input_len = len(input)
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

        bucket_index = select_bucket(ori_input_len)
        if bucket_index == -1:
          logging.warn('uttid=' + str(uttid) + ', frames is too long: '
                 + str(ori_input_len))
          continue
        caches[bucket_index][0].append(input)
        caches[bucket_index][1].append(target)
        caches[bucket_index][2] += 1

        if caches[bucket_index][2] >= batch_bucket_limit[bucket_index]:
          feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket_index][0])
          target_batch, target_batch_mask = self._create_target_batch(caches[bucket_index][1], self.dst2idx)
          # yield (feat_batch, feat_batch_mask, target_batch, target_batch_mask)
          #yield (feat_batch, target_batch, len(caches[bucket_index][0]))
          self._batch_queue.put((feat_batch, target_batch, len(caches[bucket_index][0])))
          caches[bucket_index] = [[], [], 0]
    self._put_done = True

  def get_batch(self):
    logging.info("queue size: " + str(self._batch_queue.qsize()))
    if self._batch_queue.empty() and self._put_done is True:
      return None
    while self._batch_queue.empty() and self._put_done is False:
      time.sleep(1)
      logging.info("queue size: " + str(self._batch_queue.qsize()))
    return self._batch_queue.get()

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
      # for word in (sent + [EOS]):
      #   if word is not None or word.strip() != '':
      #     x_tmp = phone2idx.get(word, UNK_INDEX)
      #     x.append(x_tmp)
      #     if x_tmp == UNK_INDEX:
      #       logging.warn('=========[ZSY]x_tmp=UNK_INDEX')
      # x = [phone2idx.get(word, UNK_INDEX) for word in (sent + [EOS])]

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

  @staticmethod
  def shuffle(list_of_files, log_dir):
    tf_os, tpath = mkstemp()
    tf = open(tpath, 'w')

    fds = [open(ff) for ff in list_of_files]

    for l in fds[0]:
      lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
      print("<CONCATE4SHUF>".join(lines), file=tf)

    [ff.close() for ff in fds]
    tf.close()

    os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

    # fnames = ['/tmp/{}.{}.shuf'.format(i, os.getpid()) for i, ff in enumerate(list_of_files)]
    fnames = [(log_dir + '/{}.{}.shuf').format(i, os.getpid()) for i, ff in enumerate(list_of_files)]
    fds = [open(fn, 'w') for fn in fnames]

    for l in open(tpath + '.shuf'):
      s = l.strip().split('<CONCATE4SHUF>')
      for i, fd in enumerate(fds):
        print(s[i], file=fd)

    [ff.close() for ff in fds]

    os.remove(tpath)
    os.remove(tpath + '.shuf')

    return fnames

  def get_test_batches_with_buckets(self, src_path, tokens_per_batch):
    buckets = [(i) for i in range(50, 10000, 10)]

    def select_bucket(sl):
      for l1 in buckets:
        if sl < l1:
          return l1
      raise Exception("The sequence is too long: ({})".format(sl))

    caches = {}
    for bucket in buckets:
      caches[bucket] = [[], [], 0]  # feats, uttids, count

    scp_reader = zark.ArkReader(src_path)
    count = 0
    while True:
      uttid, input, loop = scp_reader.read_next_utt()
      if loop:
        break

      input_len = len(input)
      bucket = select_bucket(input_len)
      caches[bucket][0].append(input)
      caches[bucket][1].append(uttid)
      caches[bucket][2] += input_len
      count = count + 1
      if caches[bucket][2] > tokens_per_batch:
        feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
        yield feat_batch, caches[bucket][1]
        caches[bucket] = [[], [], 0]

    # Clean remain sentences.
    for bucket in buckets:
      if len(caches[bucket][0]) > 0:
        logging.info('get_test_batches_with_buckets, len(caches[bucket][0])=' + str(len(caches[bucket][0])))
        feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
        yield feat_batch, caches[bucket][1]

    logging.info('get_test_batches_with_buckets, loaded count=' + str(count))

  def get_test_batches_with_buckets_and_target(self, src_path, dst_path, tokens_per_batch):
    buckets = [(i) for i in range(50, 10000, 5)]

    def select_bucket(sl):
      for l1 in buckets:
        if sl < l1:
          return l1
      raise Exception("The sequence is too long: ({})".format(sl))

    uttid_target_map = {}
    for line in codecs.open(dst_path, 'r', 'utf-8'):
      line = line.strip()
      if line == '' or line is None:
        continue
      splits = re.split('\s+', line)
      uttid = splits[0].strip()
      target = splits[1:]
      uttid_target_map[uttid] = target
    logging.info('loaded dst_path=' + str(dst_path) + ',size=' + str(len(uttid_target_map)))

    caches = {}
    for bucket in buckets:
      caches[bucket] = [[], [], 0, 0]

    scp_reader = zark.ArkReader(src_path)
    count = 0
    while True:
      uttid, input, loop = scp_reader.read_next_utt()
      if loop:
        break

      target = uttid_target_map[uttid]
      if target is None:
        logging.warn('uttid=' + str(uttid) + ',target is None')
        continue

      input_len = len(input)
      target_len = len(target)
      bucket = select_bucket(input_len)
      caches[bucket][0].append(input)
      caches[bucket][1].append(target)
      caches[bucket][2] += input_len
      caches[bucket][3] += target_len
      count = count + 1
      if caches[bucket][2] > tokens_per_batch:
        feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
        target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
        yield feat_batch, target_batch
        caches[bucket] = [[], [], 0, 0]

    for bucket in buckets:
      if len(caches[bucket][0]) > 0:
        logging.info('get_test_batches_with_buckets, len(caches[bucket][0])=' + str(len(caches[bucket][0])))
        feat_batch, feat_batch_mask = self._create_feat_batch(caches[bucket][0])
        target_batch, target_batch_mask = self._create_target_batch(caches[bucket][1], self.dst2idx)
        yield feat_batch, target_batch

    logging.info('get_test_batches_with_buckets, loaded count=' + str(count))

  def get_test_batches(self, src_path, batch_size):
    scp_reader = zark.ArkReader(src_path)
    cache = []
    uttids = []
    while True:
      uttid, feat, loop = scp_reader.read_next_utt()
      if loop:
        break
      cache.append(feat)
      uttids.append(uttid)
      if len(cache) >= batch_size:
        feat_batch, feat_batch_mask = self._create_feat_batch(cache)
        # yield feat_batch, feat_batch_mask, uttids
        yield feat_batch, uttids
        cache = []
        uttids = []
    if cache:
      feat_batch, feat_batch_mask = self._create_feat_batch(cache)
      # yield feat_batch, feat_batch_mask, uttids
      yield feat_batch, uttids

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
      sents.append(' '.join(sent))
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
