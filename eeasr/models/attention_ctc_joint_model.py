# coding=utf-8
import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops

from eeasr import model_registry
from eeasr.core import layers
from eeasr.core import utils
from eeasr.models.transformer_model import TransformerModel

@model_registry.RegisterSingleTaskModel
class AttentCTCModel(TransformerModel):
  """Model for transformer architecture."""

  def build_train_model(self, test=True, reuse=None):
    """Build model for training. """
    logging.info('Build train model.')
    self.prepare_training()

    with self.graph.as_default():
      acc_list, loss_list, gv_list = [], [], []
      ce_loss_list, ctc_loss_list = [], []
      cache = {}
      load = dict([(d, 0) for d in self._devices])
      for i, (X, Y, device) in enumerate(
            zip(self.src_pls, self.label_pls, self._devices)):

        def daisy_chain_getter(getter, name, *args, **kwargs):
          """Get a variable and cache in a daisy chain."""
          device_var_key = (device, name)
          if device_var_key in cache:
            # if we have the variable on the correct device, return it.
            return cache[device_var_key]
          if name in cache:
            # if we have it on a different device, copy it from the last device
            v = tf.identity(cache[name])
          else:
            var = getter(name, *args, **kwargs)
            v = tf.identity(var._ref())  # pylint: disable=protected-access
          # update the cache
          cache[name] = v
          cache[device_var_key] = v
          return v

        def balanced_device_setter(op):
          """Balance variables to all devices."""
          if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
            # return self._sync_device
            min_load = min(load.values())
            min_load_devices = [d for d in load if load[d] == min_load]
            chosen_device = random.choice(min_load_devices)
            load[chosen_device] += op.outputs[0].get_shape().num_elements()
            return chosen_device
          return device

        def identity_device_setter(op):
          return device

        device_setter = balanced_device_setter

        with tf.variable_scope(tf.get_variable_scope(),
                     initializer=self._initializer,
                     custom_getter=daisy_chain_getter,
                     reuse=reuse):
          with tf.device(device_setter):
            logging.info('Build model on %s.' % device)
            encoder_output = self.encoder(X, is_training=True,
                                          reuse=i > 0 or None,
                                          encoder_scope=self.encoder_scope)
            decoder_output = self.decoder(utils.shift_right(Y), encoder_output,
                                          is_training=True,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            ctc_logits, encoder_padding  = self.ctc_forward(
                encoder_output, is_training=True, reuse=i > 0 or None,
                ctc_scope='ctc_forward')
            ctc_loss = self.ctc_loss(ctc_logits, encoder_padding, label=Y,
                                     reuse=reuse, ctc_scope='CTC')

            acc, ce_loss = self.train_output(decoder_output, Y,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            ce_loss_list.append(ce_loss)
            ctc_loss_list.append(ctc_loss)
            ctc_rate = self._config.train.ctc_rate
            if not ctc_rate:
              ctc_rate = 0.1
            loss = ctc_rate * ctc_loss + (1.0 - ctc_rate) * ce_loss
            acc_list.append(acc)
            loss_list.append(loss)

            var_list = tf.trainable_variables()
            if self._config.train.var_filter:
              var_list = [v for v in var_list if 
                  re.match(self._config.train.var_filter, v.name)]
            gv_list.append(self._optimizer.compute_gradients(
                loss, var_list=var_list))

      self.accuracy = tf.reduce_mean(acc_list)
      self.loss = tf.reduce_mean(loss_list)

      # Clip gradients and then apply.
      grads_and_vars = utils.average_gradients(gv_list)
      avg_abs_grads = tf.reduce_mean(tf.abs(grads_and_vars[0]))

      if self._config.train.grads_clip > 0:
        grads, self.grads_norm = tf.clip_by_global_norm(
            [gv[0] for gv in grads_and_vars],
            clip_norm=self._config.train.grads_clip)
        grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
      else:
        self.grads_norm = tf.global_norm([gv[0] for gv in grads_and_vars])

      self.train_op = self._optimizer.apply_gradients(
          grads_and_vars, global_step=self.global_step)

      # Summaries
      tf.summary.scalar('acc', self.accuracy)
      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('ce_loss', tf.reduce_mean(ce_loss_list))
      tf.summary.scalar('ctc_loss', tf.reduce_mean(ctc_loss_list))
      tf.summary.scalar('learning_rate', self.learning_rate)
      tf.summary.scalar('grads_norm', self.grads_norm)
      tf.summary.scalar('avg_abs_grads', avg_abs_grads)
      self.summary_op = tf.summary.merge_all()

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=100)

  def ctc_forward(self, encoder_output, is_training, reuse, ctc_scope='CTC'):
    """Return CTC logist using 'encoder_output'.

    This function is called for CTC part of Attention CTC joint model.
    Args:
      encoder_output: A Tensor of [batch_size, time_step, hidden_size].
      is_training: A Boolean indicate training or eval, mainly used for dropout.
      reuse: A Boolean indicate whether reuse variable.
      ctc_scope: The scope which the function used.
    Return:
      logits: A Tensor with shape [batch_size, time_step, vocab_size+1].
      encoder_padding: A Tensor with shape [batch_size, time_step].
    """
    encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output),
                                             axis=-1), 0.0)
    with tf.variable_scope(ctc_scope, reuse=reuse):
        logits = layers.dense(encoder_output, self._config.dst_vocab_size+1,
                              use_bias=True, activation=tf.identity,
                              reuse=reuse, name='ctc_dense')
    return logits, encoder_padding

  def ctc_loss(self, logits, padding, label, reuse, ctc_scope='CTC'):
    """Return CTC loss and accurancy.

    This function calculate ctc loss and accurancy.
    Args:
      logits: A float Tensor of [batch_size, time_step, vocab_size+1].
      padding: A Boolean Tensor of [batch_size, time_step].
      label: An int32 Tensor of [batch_size, max_label_len]
      reuse: A Boolean indicate whether reuse variable.
      ctc_scope: The scope which the function used.
    Returns:
      loss: A scalar float Tensor.
    """
    src_len = tf.reduce_sum(1 - tf.cast(padding, tf.int32), 1)
    sparse_label = tf.contrib.layers.dense_to_sparse(label, 0)
    with tf.variable_scope(ctc_scope, reuse=reuse):
      loss = tf.nn.ctc_loss(labels=sparse_label, inputs=logits,
          sequence_length=src_len, time_major=False,
          ignore_longer_outputs_than_inputs=False)
    per_label_len = tf.reduce_sum(tf.cast(tf.not_equal(label, 0), tf.float32), 1)
    loss = tf.reduce_mean(loss/per_label_len)
    return loss
