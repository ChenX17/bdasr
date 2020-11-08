# coding=utf-8
from argparse import ArgumentParser
import datetime
import logging
import os
import shutil
import time
import traceback
import yaml
import random
import numpy as np
import tensorflow as tf

from eeasr import model_registry
from evaluate import Evaluator
from core import utils
from dataloader.base_dataloader import DataLoader
from dataloader.bd_dataloader import BDDataLoader
from dataloader.reverse_dataloader import REDataLoader
from dataloader import data_loader_bd as data_loader_bd
from dataloader import data_loader_read_from_scp as data_loader
from models.base_model import BaseModel
from models.transformer_model import TransformerModel
from models.bd_transformer_model import BD_TransformerModel

tf.reset_default_graph()
tf.set_random_seed(8)
random.seed(8)
np.random.seed(8)
# Future properties, using model register method,
# instead of from path import Model. Can ignore now.
print(model_registry.GetAllRegisteredClasses())

logger = logging.getLogger('')

def train(config):

  """Train a model with a config file."""
  # import pdb;pdb.set_trace()
  print('model is :', config.model)
  if 'BD' in config.model:
    data_reader = BDDataLoader(config=config)
  elif config.is_reverse:
    data_reader = REDataLoader(config=config)
    # old_data_reader = data_loader_bd.DataReader(config=config)
  else:
    data_reader = DataLoader(config=config)
    # data_reader = data_loader.DataReader(config=config)
  
  model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
  model.build_train_model(test=config.train.eval_on_dev)

  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  sess_config.allow_soft_placement = True

  summary_writer = tf.summary.FileWriter(config.model_dir, graph=model.graph)

  with tf.Session(config=sess_config, graph=model.graph) as sess:
    # Initialize all variables.
    sess.run(tf.global_variables_initializer())
    # Reload variables from checkpoint.
    if tf.train.latest_checkpoint(config.model_dir):
      available_vars = utils.available_variables(config.model_dir, config.train.from_scratch)
      if available_vars:
        saver = tf.train.Saver(var_list=available_vars)
        saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
        for v in available_vars:
          logger.info('Reload {} from disk.'.format(v.name))
      else:
        logger.info('Nothing to be reload from disk.')
    else:
      logger.info('Nothing to be reload from disk.')

    def train_one_step(batch, old_batch=None):
      (feat_batch, target_batch, batch_size) = batch
      #(old_feat_batch, old_target_batch, old_batch_size) = old_batch
      logger.info("feat_batch_size: " + str(feat_batch.shape)
              + " label_batch_size: " + str(target_batch.shape))
      feed_dict = data_reader.expand_feed_dict(
              {model.src_pls: feat_batch,
               model.label_pls: target_batch})
      step, lr, loss, _ = sess.run(
        [model.global_step, model.learning_rate, model.loss, model.train_op],
        feed_dict=feed_dict)
      if step % config.train.summary_freq == 0:
        summary = sess.run(model.summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary, global_step=step)
      return step, lr, loss

    def maybe_save_model(config):
      mp = config.model_dir + '/model_epoch_{}'.format(epoch)
      model.saver.save(sess, mp)
      logger.info('Save model in %s.' % mp)
      
      if config.train.eval_on_dev:
        evaluator = Evaluator(config)
        evaluator.init_from_existed(config, model, sess)
        evaluator.translate(config.dev.feat_file_pattern,
                            config.dev.output_file + 'decode_result_epoch_' + '{}'.format(str(epoch)))

    step = 0
    if config.train.start_epoch is None:
      start_epoch = 1
    else:
      start_epoch = config.train.start_epoch+1
    for epoch in range(start_epoch, config.train.num_epochs+1):
      data_reader.reset()
      # old_data_reader.reset()
      start_time_data_loader = time.time()
      while True:
        batch = data_reader.get_batch()
        # old_batch = old_data_reader.get_batch()
        if batch == None:
          break
        logger.info('data_load time: %.4f' %
                    (time.time() - start_time_data_loader))
        # Train normal instances.
        start_time = time.time()
        step, lr, loss = train_one_step(batch)
        if step > 1550:
          break
        logger.info(
            'epoch: %d\tstep: %d\tlr: %.6f\tloss: %.4f'
            '\ttime: %.4f\tbatch_size: %d' %
            (epoch, step, lr, loss, time.time() - start_time, batch[2]))
        # Save model
        if config.train.save_freq > 0 and step % config.train.save_freq == 0:
          maybe_save_model(config=config)

        # Stop training.
        if config.train.max_steps and step >= config.train.max_steps:
          break
        start_time_data_loader = time.time()

      # Save model per epoch if config.train.save_freq is less or equal than zero.
      if config.train.save_freq <= 0:
        maybe_save_model(config=config)

    logger.info("Finish training.")

def config_logging(log_file):
  logging.basicConfig(filename=log_file, level=logging.INFO,
      format='%(asctime)s %(filename)s[line:%(lineno)d] '
      '%(levelname)s %(message)s',
      datefmt='%a, %d %b %Y %H:%M:%S', filemode='w')
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter(
      '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


if __name__ == '__main__':
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

  parser = ArgumentParser()
  parser.add_argument('-c', '--config', dest='config')
  parser.add_argument('-d', '--delete_old_dir', dest='delete_old_dir',
      action='store_true')
  args = parser.parse_args()
  # Default config.
  if not args.config:
    args.config = './eeasr/config/my_try.yaml'
  config = utils.AttrDict(yaml.load(open(args.config)))
  if os.path.exists(config.model_dir):
    if args.delete_old_dir:
      shutil.rmtree(config.model_dir)
  # Logger
  if not os.path.exists(config.model_dir):
    os.makedirs(config.model_dir)
  config_logging(config.model_dir + '/train.log')
  shutil.copy(args.config, config.model_dir)
  time_stamp = datetime.datetime.now()
  logger.info("Training start_time: " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
  try:
    train(config)
  except Exception, e:
    logger.error(traceback.format_exc())
  time_stamp = datetime.datetime.now()
  logger.info("Training end_time:" + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))

