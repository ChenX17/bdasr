# coding=utf-8
import logging
import random
import re

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import init_ops

from eeasr import model_registry
from eeasr.core import layers
from eeasr.core import layers_with_attention
from eeasr.core.layers import embedding, residual, dense, ff_hidden
from eeasr.core.layers_with_attention import multihead_attention
from eeasr.core.utils import average_gradients, shift_right
from eeasr.core.utils import learning_rate_decay
from eeasr.models.base_model import BaseModel
def sparse_tuple_from(sequences, dtype=np.int32):
  """Creates a sparse representention of ``sequences``.
  Args: 
    * sequences: a list of lists of type dtype where each element is a sequence
      Returns a tuple with (indices, values, shape)
    """
  indices = []
  values = []

  for n, seq in enumerate(sequences):
    indices.extend(zip([n]*len(seq), range(len(seq))))
    values.extend(seq)

  indices = np.asarray(indices, dtype=np.int64)
  values = np.asarray(values, dtype=dtype)
  shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int64)

  return tf.SparseTensor(indices=indices, values=values, shape=shape) 
  #return indices, values, shape

@model_registry.RegisterSingleTaskModel
class CTC_joint_TransformerModel(BaseModel):
  """Model for transformer architecture."""

  def __init__(self, *args, **kargs):
    super(CTC_joint_TransformerModel, self).__init__(*args, **kargs)
    activations = {"relu": tf.nn.relu,
                   "sigmoid": tf.sigmoid,
                   "tanh": tf.tanh,
                   "swish": lambda x: x * tf.sigmoid(x),
                   "glu": lambda x, y: x * tf.sigmoid(y)}
    #self.seq_len = tf.placeholder(tf.int32, [None])
    with self.graph.as_default():
      src_pls = []
      label_pls = []
      #x_len_pls = []
      #y_len_pls = []
      for i, device in enumerate(self._devices):
        with tf.device(device):
          pls_batch_x = tf.placeholder(
                  dtype=tf.float32,
                  shape=[None, None, self._config.train.input_dim],
                  name='src_pl_{}'.format(i))
          pls_batch_y = tf.placeholder(
                  dtype=tf.int32, shape=[None, None],
                  name='dst_pl_{}'.format(i))
          #pls_x_len = tf.placeholder(
                  #tf.int32,
                  #name='x_len_pl_{}'.format(i))
          #pls_y_len = tf.placeholder(
                 # tf.int32,
                  #name='y_len_pl_{}'.format(i))
          src_pls.append(pls_batch_x)
          label_pls.append(pls_batch_y)
          #x_len_pls.append(pls_x_len)
          #y_len_pls.append(pls_y_len)
      self.src_pls = tuple(src_pls)
      self.label_pls = tuple(label_pls)
      #self.x_len_pls = tuple(x_len_pls)
      #self.y_len_pls = tuple(y_len_pls)

    self._ff_activation = activations[self._config.ff_activation]
  def build_train_model(self, test=True, reuse=None):
    """Build model for training. """
    logging.info('Build train model.')
    self.prepare_training()

    with self.graph.as_default():
      acc_list, loss_list, gv_list = [], [], []
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
            XL = tf.reduce_sum(1-tf.cast(tf.equal(tf.reduce_sum(tf.abs(encoder_output),axis=-1),0),tf.int32),1)
            ctc_output = dense(encoder_output, self._config.dst_vocab_size+1, activation=tf.identity, use_bias=True, name='ctc_linear')
            #ctc_output = tf.transpose(ctc_output, (1,0,2))
            #YL =  Y.get_shape()[-1]
            #XL = Y.get_shape()[-2]

            SY = tf.contrib.layers.dense_to_sparse(Y, eos_token=3)
            ctc_loss = tf.nn.ctc_loss(SY, inputs=ctc_output, sequence_length=XL, time_major=False, ignore_longer_outputs_than_inputs=False)
            decoder_output = self.decoder(Y, encoder_output,
                                          is_training=True,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            acc, loss = self.train_output(decoder_output, Y,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            lambda_x = 0.2
            loss = lambda_x * ctc_loss + (1 - lambda_x) * loss
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
      grads_and_vars = average_gradients(gv_list)
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
      tf.summary.scalar('learning_rate', self.learning_rate)
      tf.summary.scalar('grads_norm', self.grads_norm)
      tf.summary.scalar('avg_abs_grads', avg_abs_grads)
      self.summary_op = tf.summary.merge_all()

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=100)

    # We may want to test the model during training.
    if test:
      self.build_test_model(reuse=True)
  def encoder_impl(self, encoder_input, is_training):

    attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
    residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

    # Mask
    encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_input), axis=-1), 0.0)
    encoder_output = dense(encoder_input, self._config.hidden_units, activation=tf.identity,
                 use_bias=True, name="src_change")
    #encoder_output = tf.contrib.layers.layer_norm(encoder_output, center=True, scale=True, trainable=True)
    encoder_output = layers.layer_norm(encoder_output)

    # Add positional signal
    encoder_output = layers_with_attention.add_timing_signal_1d(encoder_output)
    # Dropout
    encoder_output = tf.layers.dropout(encoder_output,
                       rate=residual_dropout_rate,
                       training=is_training)

    # Blocks
    for i in range(self._config.encoder_num_blocks):
      with tf.variable_scope("block_{}".format(i)):
        # Multihead Attention
        encoder_output = residual(encoder_output,
                      multihead_attention(
                        query_antecedent=encoder_output,
                        memory_antecedent=None,
                        bias=layers_with_attention.attention_bias_ignore_padding(encoder_padding),
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        name='encoder_self_attention',
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Feed Forward
        encoder_output = residual(encoder_output,
                      ff_hidden(
                        inputs=encoder_output,
                        hidden_size=4 * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)
    # Mask padding part to zeros.
    encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
    return encoder_output

  def decoder_impl(self, decoder_input, encoder_output, is_training):
    # decoder_input: [batch_size, step]
    # encoder_output: [batch_size, time_step, hidden_units]
    attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
    residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

    encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
    encoder_attention_bias = layers_with_attention.attention_bias_ignore_padding(encoder_padding)

    decoder_output = embedding(decoder_input,
                   vocab_size=self._config.dst_vocab_size,
                   dense_size=self._config.hidden_units,
                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                   name="dst_embedding")
    # Positional Encoding
    # decoder_output += layers_with_attention.add_timing_signal_1d(decoder_output)
    decoder_output = layers_with_attention.add_timing_signal_1d(decoder_output)
    # Dropout
    decoder_output = tf.layers.dropout(decoder_output,
                       rate=residual_dropout_rate,
                       training=is_training)
    # Bias for preventing peeping later information
    self_attention_bias = layers_with_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])

    # Blocks
    for i in range(self._config.decoder_num_blocks):
      with tf.variable_scope("block_{}".format(i)):
        # Multihead Attention (self-attention)
        decoder_output = residual(decoder_output,
                      multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=None,
                        bias=self_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        output_depth=self._config.hidden_units,
                        name="decoder_self_attention",
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Multihead Attention (vanilla attention)
        decoder_output = residual(decoder_output,
                      multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=encoder_output,
                        bias=encoder_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        name="decoder_vanilla_attention",
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Feed Forward
        decoder_output = residual(decoder_output,
                      ff_hidden(
                        decoder_output,
                        hidden_size=4 * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)
    return decoder_output

  def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):
    # decoder_input: [batch_size * beam_size, step], 该step逐步增加，即1,2,3,..
    # decoder_cache: [batch_size * beam_size, 0, num_blocks , hidden_units ]
    # encoder_output: [batch_size * beam_size, time_step, hidden_units]
    attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
    residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

    encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
    encoder_attention_bias = layers_with_attention.attention_bias_ignore_padding(encoder_padding)

    decoder_output = embedding(decoder_input,
                   vocab_size=self._config.dst_vocab_size,
                   dense_size=self._config.hidden_units,
                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                   name="dst_embedding")
    # Positional Encoding
    # decoder_output += layers_with_attention.add_timing_signal_1d(decoder_output)
    decoder_output = layers_with_attention.add_timing_signal_1d(decoder_output)
    # Dropout
    decoder_output = tf.layers.dropout(decoder_output,
                       rate=residual_dropout_rate,
                       training=is_training)

    new_cache = []

    # Blocks
    for i in range(self._config.decoder_num_blocks):
      with tf.variable_scope("block_{}".format(i)):
        # Multihead Attention (self-attention)
        decoder_output = residual(decoder_output[:, -1:, :],
                      multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=None,
                        bias=None,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        reserve_last=True,
                        output_depth=self._config.hidden_units,
                        name="decoder_self_attention",
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Multihead Attention (vanilla attention)
        decoder_output = residual(decoder_output,
                      multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=encoder_output,
                        bias=encoder_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        reserve_last=True,
                        name="decoder_vanilla_attention",
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Feed Forward
        decoder_output = residual(decoder_output,
                      ff_hidden(
                        decoder_output,
                        hidden_size=4 * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)

        decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
        new_cache.append(decoder_output[:, :, None, :])

    new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

    return decoder_output, new_cache
