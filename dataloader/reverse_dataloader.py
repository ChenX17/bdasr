# coding=utf-8

'''
Date: 2020-10-29 20:43:54
LastEditors: Xi Chen(chenxi50@lenovo.com)
LastEditTime: 2020-11-07 20:39:25
'''

import random
import logging
import kaldi_io
import numpy as np
import six.moves.queue as queue
import codecs
import glob

from base_dataloader import DataLoader
import data_augmentation

PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 3

PAD = u'<PAD>'
UNK = u'<UNK>'
EOS = u'</S>'

class REDataLoader(DataLoader):
    def __init__(self, config):
        super(REDataLoader, self).__init__(config=config)
        #self._config = config
    def _create_target_batch(self, sents, phone2idx):
        # sents的数据是[word1 word2 ... wordn]
        indices = []
        for sent in sents:
            x = []

            for word in (sent + [EOS]):
                x_tmp = phone2idx.get(word, UNK_INDEX)
                x.append(x_tmp)
                if x_tmp == UNK_INDEX and word != UNK:
                    logging.warn('x_tmp=UNK_INDEX, word=' + str(word.encode('UTF-8')))
            indices.append(x)

        # Pad to the same length.
        batch_size = len(sents)
        maxlen = max([len(s) for s in indices])
        target_batch = np.zeros([batch_size, maxlen], np.int32)
        target_batch.fill(PAD_INDEX)
        target_batch_mask = np.ones([batch_size, maxlen], dtype=np.int32)
        for i, x in enumerate(indices):
            target_batch[i, :len(x)] = x[:-1][::-1]+[3]
            target_batch_mask[i, :len(x)] = 0
        return target_batch