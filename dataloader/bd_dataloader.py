# coding=utf-8
import random
import logging
import kaldi_io
import numpy as np
import six.moves.queue as queue
import codecs
import glob

from base_dataloader import DataLoader
import data_augmentation

random.seed(8)
np.random.seed(8)

PAD_INDEX = 0
UNK_INDEX = 1
EOS_INDEX = 3

PAD = u'<PAD>'
UNK = u'<UNK>'
EOS = u'</S>'

class BDDataLoader(DataLoader):
    def __init__(self,config):
        super(BDDataLoader, self).__init__(config=config)
        # # Variable init.
        # self._put_done = False
        # self._batch_queue = queue.Queue(100)
        # self.bucket_select_dict = {}
        # self.batch_bucket_limit = []
        # self.uttid_target_map = {}

        # # Config init.
        # self._config = config
        # self.apply_sentence_cmvn = self._config.apply_sentence_cmvn
        # self.global_cmvn_file = self._config.global_cmvn_file
        # self.global_cmvn = self._maybe_load_mean_stddev(self.global_cmvn_file)

        # self.feat_files = glob.glob(self._config.train.feat_file_pattern)
        # self.scp_files = glob.glob(self._config.train.feat_file_pattern)
        # self.label_file = self._config.train.label_file
        # self.frame_bucket_limit = self._config.train.frame_bucket_limit
        # self.frame_bucket_limit = self.frame_bucket_limit\
        #     .replace('[','')\
        #     .replace(']','')\
        #     .replace(' ','')
        # self.frame_bucket_limit = [int(i) for i in 
        #     self.frame_bucket_limit.split(',')]
        # self.batch_bucket_limit_per_gpu = \
        #     self._config.train.batch_bucket_limit_per_gpu
        # self.batch_bucket_limit_per_gpu = self.batch_bucket_limit_per_gpu\
        #     .replace('[','')\
        #     .replace(']','')\
        #     .replace(' ','')
        # self.batch_bucket_limit_per_gpu = [int(int(i)*self._config.train.batch_factor) for i in self.batch_bucket_limit_per_gpu.split(',')]
        # logging.info('frame_bucket_limit: ' + str(self.frame_bucket_limit))
        # logging.info('batch_bucket_limit_per_gpu: ' + str(self.batch_bucket_limit_per_gpu))

        # # Function init.
        # self.load_vocab_init()
        # self.select_bucket_init()
        # self.load_label_init()


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
                if uttid.split('-')[0]=='0.9' or uttid.split('-')[0]=='1.1':
                    continue
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
                self._batch_queue.put((feat_batch, target_batch, len(caches[bucket_index][0])))
                caches[bucket_index] = [[], [], 0]
        os.remove(scp_shuf_path)
        self._put_done = True
        del(caches)

    def _create_bd_target_batch(self, sents, phone2idx):
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
        target_batch_l2r = np.zeros([batch_size, maxlen], np.int32)
        target_batch_r2l = np.zeros([batch_size, maxlen], np.int32)
        target_batch_l2r.fill(PAD_INDEX)
        target_batch_r2l.fill(PAD_INDEX)
        for i, x in enumerate(indices):
            target_batch_l2r[i, :len(x)] = x
            target_batch_r2l[i, :len(x)] = x[:-1][::-1]+[3]

        target_batch_l2r = target_batch_l2r[:, np.newaxis, :]
        target_batch_r2l = target_batch_r2l[:, np.newaxis, :]
        btarget = np.concatenate((target_batch_l2r, target_batch_r2l), 1)

        return btarget

