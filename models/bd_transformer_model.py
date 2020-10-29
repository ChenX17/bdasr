# coding=utf-8
import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python.util import nest
from eeasr import model_registry
from eeasr.core import layers
from eeasr.core import layers_with_attention
from eeasr.core import utils
from eeasr.core.layers import embedding, residual, dense, ff_hidden
from eeasr.core.layers_with_attention import multihead_attention
from eeasr.core.layers_with_attention import sb_multihead_attention_for_decoding
from eeasr.core.layers_with_attention import sb_multihead_attention
from eeasr.core.utils import average_gradients, shift_right
from eeasr.core.utils import learning_rate_decay
from eeasr.models.base_model import BaseModel
import copy
@model_registry.RegisterSingleTaskModel
class BD_TransformerModel(BaseModel):
  """Model for transformer architecture."""

  def __init__(self, config, num_gpus, *args, **kargs):
    super(BD_TransformerModel, self).__init__(config, num_gpus, *args, **kargs)
    self.graph = tf.Graph()
    self._config = config
    # global is_attention_smoothing
    # is_attention_smoothing = self._config.is_attention_smoothing
    # logging.info('[ZSY_INFO]is_attention_smoothing='
    #     + str(is_attention_smoothing))
    # logging.info('[ZSY_INFO]is_lsoftmax=' + str(self._config.is_lsoftmax))
    self._devices = ['/gpu:%d' % i for i in
        range(num_gpus)] if num_gpus > 0 else ['/cpu:0']

    # Placeholders and saver.
    with self.graph.as_default():
      src_pls = []
      label_pls = []
      for i, device in enumerate(self._devices):
        with tf.device(device):
          # Model X inputs: [batch, feat, feat_dim]
          pls_batch_x = tf.placeholder(
              dtype=tf.float32,
              shape=[None, None, self._config.train.input_dim],
              name='src_pl_{}'.format(i))
          # Model Y inputs: [batch, len]
          pls_batch_y = tf.placeholder(
              dtype=tf.int32, shape=[None, 2, None],
              name='dst_pl_{}'.format(i))
          src_pls.append(pls_batch_x)
          label_pls.append(pls_batch_y)
      self.src_pls = tuple(src_pls)
      self.label_pls = tuple(label_pls)

    self.encoder_scope = 'encoder'
    self.decoder_scope = 'decoder'

    activations = {"relu": tf.nn.relu,
                   "sigmoid": tf.sigmoid,
                   "tanh": tf.tanh,
                   "swish": lambda x: x * tf.sigmoid(x),
                   "glu": lambda x, y: x * tf.sigmoid(y)}
    self._ff_activation = activations[self._config.ff_activation]

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
                        hidden_size=self._config.ff_units * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)
    # Mask padding part to zeros.
    encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
    return encoder_output

  def decoder_impl(self, decoder_input, encoder_output, is_training, cache=None):
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
    #decoder_output = layers_with_attention.bd_add_timing_signal_1d(decoder_output)
    decoder_output = tf.concat(
      [tf.expand_dims(layers_with_attention.add_timing_signal_1d(decoder_output[0]), 0),
      tf.expand_dims(layers_with_attention.add_timing_signal_1d(decoder_output[1]), 0)], 0)
    # Dropout
    decoder_output = tf.layers.dropout(decoder_output,
                       rate=residual_dropout_rate,
                       training=is_training)
    # Bias for preventing peeping later information
    # Bias for preventing peeping later information for bidirectional decoder
    self_attention_bias = layers_with_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[2])

    # Blocks
    for i in range(self._config.decoder_num_blocks):
      with tf.variable_scope("block_{}".format(i)):
        layer_name = "layer_%d" % i
        layer_cache = cache[layer_name] if cache is not None else None
        # Multihead Attention (self-attention)
        decoder_output = residual(decoder_output,
                      sb_multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=None,
                        cache=layer_cache,
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
                      sb_multihead_attention(
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
                        hidden_size=self._config.ff_units * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)
    return decoder_output

  def decoder_with_caching(self, decoder_input, decoder_cache,
    encoder_output, is_training, reuse, cache_qkv, decoder_scope):
    """Incremental Decoder"""
    with tf.variable_scope(decoder_scope, reuse=reuse):
      return self.decoder_with_caching_impl(
          decoder_input, decoder_cache, encoder_output, is_training, cache_qkv)

  def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training, cache_qkv=None):
    # decoder_input: [batch_size * beam_size, step], 该step逐步增加，即1,2,3,..
    # decoder_cache: [batch_size * beam_size, 0, num_blocks , hidden_units ]
    # encoder_output: [batch_size * beam_size, time_step, hidden_units]
    batch_size = tf.shape(encoder_output)[0]/self._config.test.beam_size
    attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
    residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0
    encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
    encoder_attention_bias = layers_with_attention.attention_bias_ignore_padding(encoder_padding)
    decoder_self_attention_bias = layers_with_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])
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
        layer_name = "layer_%d" % i
        layer_cache = cache_qkv[layer_name] if cache_qkv is not None else None
        # Multihead Attention (self-attention)
        decoder_output = residual(decoder_output[:,-1:,:],
                      sb_multihead_attention_for_decoding(
                        query_antecedent=decoder_output[:,-1:,:],
                        memory_antecedent=None,
                        cache=layer_cache,
                        bias=decoder_self_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        batch_size=batch_size,
                        beam_size = self._config.test.beam_size,
                        output_depth=self._config.hidden_units,
                        name="decoder_self_attention",
                        summaries=True),
                      dropout_rate=residual_dropout_rate)
        # Multihead Attention (vanilla attention)
        multihead_out = sb_multihead_attention_for_decoding(
                        query_antecedent=decoder_output,
                        memory_antecedent=encoder_output,
                        bias=encoder_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        name="decoder_vanilla_attention",
                        summaries=True)
        decoder_output = residual(decoder_output, multihead_out,
                      dropout_rate=residual_dropout_rate)

        # Feed Forward
        decoder_output = residual(decoder_output,
                      ff_hidden(
                        decoder_output,
                        hidden_size=self._config.ff_units * self._config.hidden_units,
                        output_size=self._config.hidden_units,
                        activation=self._ff_activation),
                      dropout_rate=residual_dropout_rate)
        decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
        new_cache.append(decoder_output[:, :, None, :])
    new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

    return decoder_output, new_cache, cache_qkv
  
  def build_train_model(self, test=True, reuse=None):
    """Build model for training. """
    logging.info('Build train model.')
    self.prepare_training()

    with self.graph.as_default():
      acc_list = []
      loss_list = []
      l2r_loss_list = []
      r2l_loss_list = []
      gv_list = []
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
            Y = tf.transpose(Y, perm=[1, 0, 2])
            decoder_output = self.decoder(utils.shift(Y), encoder_output,
                                          is_training=True,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            acc, loss, l2r_loss, r2l_loss = self.train_output(decoder_output, Y,
                                             reuse=i > 0 or None,
                                             decoder_scope=self.decoder_scope)
            var_list = tf.trainable_variables()
            if self._config.train.var_filter:
              var_list = [v for v in var_list if
                  re.match(self._config.train.var_filter, v.name)]
            #l2_loss_list = []
            #for i in var_list:
            #  l2_loss_list.append(self.l2_weight * tf.nn.l2_loss(i))
            #l2_loss = tf.reduce_sum(l2_loss_list)
            #loss += l2_loss
            acc_list.append(acc)
            loss_list.append(loss)
            l2r_loss_list.append(l2r_loss)
            r2l_loss_list.append(r2l_loss)

            gv_list.append(self._optimizer.compute_gradients(
                loss, var_list=var_list))

      self.accuracy = tf.reduce_mean(acc_list)
      self.loss = tf.reduce_mean(loss_list)
      self.l2r_loss = tf.reduce_mean(l2r_loss_list)
      self.r2l_loss = tf.reduce_mean(r2l_loss_list)

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
      tf.summary.scalar('l2r_loss', self.l2r_loss)
      tf.summary.scalar('r2l_loss', self.r2l_loss)
      #tf.summary.scalar('l2_loss', l2_loss)
      tf.summary.scalar('learning_rate', self.learning_rate)
      tf.summary.scalar('grads_norm', self.grads_norm)
      tf.summary.scalar('avg_abs_grads', avg_abs_grads)
      self.summary_op = tf.summary.merge_all()

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=60)

    # We may want to test the model during training.
    if test:
      self.build_test_model(reuse=True)
  
  def train_output(self, decoder_output, Y, reuse, decoder_scope):
    """Calculate loss and accuracy."""
    with tf.variable_scope(decoder_scope, reuse=reuse):
      if self._config.is_lsoftmax is None:
        self._config.is_lsoftmax = False
      if not self._config.is_lsoftmax:
        logits = layers.dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
                 name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                 reuse=True if self._config.tie_embedding_and_softmax else None)
      else:
        with tf.variable_scope("dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                     "dense", reuse=reuse):
          input_size = decoder_output.get_shape().as_list()[-1]
          inputs_shape = tf.unstack(tf.shape(decoder_output))
          decoder_output_tmp = tf.reshape(decoder_output, [-1, input_size])
          Y_tmp = tf.reshape(Y, [-1])
          with tf.variable_scope(tf.get_variable_scope(),
                       reuse=True if self._config.tie_embedding_and_softmax else None):
            weights = tf.get_variable("kernel", [self._config.dst_vocab_size, input_size])
            weights = tf.transpose(weights)
            logits = lsoftmax(decoder_output_tmp, weights, Y_tmp)
            logits = tf.reshape(logits, inputs_shape[:-1] + [self._config.dst_vocab_size])

      preds = tf.to_int32(tf.argmax(logits, axis=-1))
      l2r_preds = preds[0]
      r2l_preds = preds[1]

      mask_l2r = tf.to_float(tf.not_equal(Y[0], 0))
      mask_r2l = tf.to_float(tf.not_equal(Y[1], 0))
      l2r_acc = tf.reduce_sum(tf.to_float(tf.equal(l2r_preds, Y[0])) * mask_l2r) / tf.reduce_sum(mask_l2r)
      r2l_acc = tf.reduce_sum(tf.to_float(tf.equal(r2l_preds, Y[1])) * mask_r2l) / tf.reduce_sum(mask_r2l)
      #acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)
      acc = (l2r_acc + r2l_acc)/2
      # Smoothed loss
      l2r_loss = layers.smoothing_cross_entropy(logits=logits[0], labels=Y[0],
                             vocab_size=self._config.dst_vocab_size,
                             confidence=1 - self._config.train.label_smoothing)
      r2l_loss = layers.smoothing_cross_entropy(logits=logits[1], labels=Y[1],
                             vocab_size=self._config.dst_vocab_size,
                             confidence=1 - self._config.train.label_smoothing)
      mean_l2r_loss = tf.reduce_sum(l2r_loss*mask_l2r) / (tf.reduce_sum(mask_l2r))
      mean_r2l_loss = tf.reduce_sum(r2l_loss*mask_r2l) / (tf.reduce_sum(mask_r2l))
      mean_loss = (mean_l2r_loss + mean_r2l_loss) / 2

    return acc, mean_loss, mean_l2r_loss, mean_r2l_loss
  
  def build_test_model(self, reuse=None):
    """Build model for inference."""
    logging.info('Build test model.')
    with self.graph.as_default(), tf.variable_scope(
        tf.get_variable_scope(), reuse=reuse):
      prediction_list = []
      atten_probs_list = []
      scores_list = []
      alive_probs_list = []
      finished_flags_list = []
      loss_sum = 0
      for i, (X, Y, device) in enumerate(
          zip(self.src_pls, self.label_pls, self._devices)):
        with tf.device(device):
          logging.info('Build model on %s.' % device)
          dec_input = utils.shift(Y)
          #dec_input = Y

          # Avoid errors caused by empty input by a condition phrase.
          def true_fn():
            enc_output = self.encoder(X, is_training=False,
                reuse=i > 0 or None, encoder_scope=self.encoder_scope)
            prediction, scores, alive_probs, finished_flags = self.beam_search(enc_output, vocab_size=self._config.vocab_size, reuse=i > 0 or None)
            #dec_output = self.decoder(dec_input, enc_output, is_training=False,
                #reuse=True, decoder_scope=self.decoder_scope)
            #loss = self.test_loss(dec_output, Y, reuse=True,
                #decoder_scope=self.decoder_scope)
            return prediction, scores, alive_probs, finished_flags

          def false_fn():
            return tf.zeros([0, 0], dtype=tf.int32), tf.zeros([0, 0], dtype=tf.float32), tf.zeros([0, 0], dtype=tf.float32), tf.zeros([0, 0], dtype=tf.bool)
            #return tf.zeros([0, 0, 0], dtype=tf.int32)

          prediction, scores, alive_probs, finished_flags = tf.cond(tf.greater(tf.shape(X)[0], 0),
                                      true_fn, false_fn)
                                  
          prediction_list.append(prediction)
          scores_list.append(scores)
          alive_probs_list.append(alive_probs)
          finished_flags_list.append(finished_flags)

      max_length = tf.reduce_max(
          [tf.shape(pred)[1] for pred in prediction_list])

      def pad_to_max_length(input, length):
        """Pad the input (with rank 2) with 3(</S>)
        to the given length in the second axis."""
        shape = tf.shape(input)
        padding = tf.ones([shape[0], length - shape[1]], dtype=tf.int32) * 3
        return tf.concat([input, padding], axis=1)

      prediction_list = [pad_to_max_length(pred, max_length)
          for pred in prediction_list]
      self.prediction = tf.concat(prediction_list, axis=0)
      self.scores = tf.concat(scores_list, axis=0)
      self.alive_probs = tf.concat(alive_probs_list, axis=0)
      self.finished_flags = tf.concat(finished_flags_list, axis=0)
      #self.prediction = prediction_list
      self.atten_probs = atten_probs_list
      self.loss_sum = loss_sum

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=60)
  
  def beam_search(self, encoder_output, vocab_size, reuse, decode_length=50):
    """Beam search in graph."""
    beam_size = self._config.test.beam_size
    #batch_size = tf.shape(encoder_output)[0]\
    batch_size =  shape_list(encoder_output)[0]
    #batch_size = self._config.test.batch_size
    inf = 1e10
    cache_qkv = {
                "layer_%d" % layer:{
                "k": tf.zeros([batch_size, 0, self._config.hidden_units]),
                "v": tf.zeros([batch_size, 0, self._config.hidden_units]),}
                for layer in range(self._config.decoder_num_blocks)}
    for layer in cache_qkv:
      cache_qkv[layer]["k"].set_shape = tf.TensorShape([None, None, self._config.hidden_units])
      cache_qkv[layer]["v"].set_shape = tf.TensorShape([None, None, self._config.hidden_units])          
  
    #innitial log probs [1,beam_size]
    # [:beam/2] shows the probs started with <l2r>
    # [beam/2:] shows the probs started with <r2l>
    initial_log_probs = tf.constant([[0.] + (int(beam_size/2)-1)*[-inf] + [0.] + (int(beam_size/2)-1)*[-inf]])
    #[batch_size, beam_size]
    alive_log_probs = tf.tile(initial_log_probs, [batch_size, 1])
    # [batch, 1] index 2 is <l2r>
    initial_ids_1 = 2*tf.ones([batch_size,1], dtype=tf.int32)
    # [batch, 1] index 4 is <r2l>
    initial_ids_2 = 4*tf.ones([batch_size,1], dtype=tf.int32)
    # [batch, beam/2, 1]
    alive_seq_1 = tf.tile(tf.expand_dims(initial_ids_1, 1), [1, tf.cast(beam_size/2, tf.int32), 1])
    alive_seq_2 = tf.tile(tf.expand_dims(initial_ids_2, 1), [1, tf.cast(beam_size/2, tf.int32), 1])
    # [batch, beam, 1]
    alive_seq = tf.concat([alive_seq_1, alive_seq_2], axis=1)

    flat_curr_scores = tf.zeros(shape_list(alive_seq), tf.float32)
    # [batch, beam, 1] show the sentences decodered
    #finished_seq = tf.ones(shape_list(alive_seq), tf.int32)*3
    finished_seq_1 = tf.tile(tf.expand_dims(initial_ids_1, 1), [1, tf.cast(beam_size/2, tf.int32), 1])
    finished_seq_2 = tf.tile(tf.expand_dims(initial_ids_2, 1), [1, tf.cast(beam_size/2, tf.int32), 1])
    finished_seq = tf.concat([finished_seq_1, finished_seq_2], axis=1)
    # [batch, beam]
    finished_scores = tf.ones([batch_size, beam_size]) * (-inf)
    # [batch, beam] show the sentences is finished or not
    finished_flags = tf.zeros([batch_size, beam_size], tf.bool)
    
    # Prepare beam search inputs.
    # [batch_size, 1, feat_len, hidden_units]
    encoder_output = encoder_output[:, None, :, :]
    # [batch_size, beam_size, feat_len, hidden_units]
    encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
    # [batch_size * beam_size, feat_len, hidden_units]
    encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1,
        encoder_output.get_shape()[-1].value])
    cache_qkv = nest.map_structure(
                lambda t: _expand_to_beam_size(t, beam_size), cache_qkv)
    states = tf.zeros([batch_size, beam_size, 0,
                      self._config.decoder_num_blocks,
                      self._config.hidden_units])

    def step(flat_curr_scores, i, alive_seq, alive_log_probs, finished_seq,
     finished_scores, finished_flags, states, cache_qkv):
      # Where are we.
      i += 1
      flat_ids = tf.reshape(alive_seq, [batch_size * beam_size, -1])
      # Call decoder and get predictions.
      flat_states = _merge_beam_dim(states)
      flat_cache = nest.map_structure(_merge_beam_dim, cache_qkv)
      decoder_output, flat_states, flat_cache = self.decoder_with_caching(
          flat_ids, flat_states, encoder_output, cache_qkv=flat_cache, is_training=False,
          reuse=reuse, decoder_scope='decoder')
      logits = self.test_output(
          decoder_output, reuse=reuse, decoder_scope='decoder')
      logits = tf.reshape(logits, [batch_size, beam_size, -1])
      # softmax
      candidate_log_probs = logits - tf.reduce_logsumexp(logits, axis=2, keep_dims=True)

      log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)
      length_penalty = tf.pow(((5. + tf.to_float(i + 1)) / 6.), self._config.test.lp_alpha)
      curr_scores = log_probs / length_penalty
      flat_curr_scores = tf.reshape(curr_scores, [batch_size, 2, -1]) # (batch, 2, beam/2*vocab)
      topk_scores, topk_ids = tf.nn.top_k(flat_curr_scores, k=beam_size, sorted=True) ## [batch, 2, beam]
      topk_log_probs = topk_scores * length_penalty
      topk_log_probs = tf.reshape(topk_log_probs, [-1, 2*beam_size])

      topk_scores = tf.reshape(topk_scores, [-1, 2*beam_size]) ## add;
      #topk_beam_index is the beam index of topk
      topk_beam_index = topk_ids // self._config.dst_vocab_size ## like [[0,1,1,0],[1,1,0,0],[1,0,0,0],...], e.g. beam=2
      #topk_ids is the class label of topk
      topk_ids %= self._config.dst_vocab_size  # Unflatten the ids
      #[bacth, 2, beam]
      ##add;
      topk_beam_index_1 = tf.concat([tf.expand_dims(topk_beam_index[:,0,:],1), tf.expand_dims(topk_beam_index[:,1,:]+tf.cast(beam_size/2,tf.int32),1)], axis=1)
      topk_beam_index = tf.reshape(topk_beam_index_1, [-1, beam_size*2])
      topk_ids = tf.reshape(topk_ids, [-1, beam_size*2])

      batch_pos = compute_batch_indices(batch_size, beam_size * 2) # like [[0,0,0,0,],[1,1,1,1],[2,2,2,2],...] (batch, 2*beam) 
      topk_coordinates = tf.stack([batch_pos, topk_beam_index], axis=2) # like [[[0,0],[0,1],[0,1],[0,0]], [[1,1],[1,1],[1,0],[1,0]], [[2,1],[2,0],[2,0],[2,0]],...]  (batch, 2*beam, 2)
      topk_seq = tf.gather_nd(alive_seq, topk_coordinates) # (batch, 2*beam, lenght)
      states = _unmerge_beam_dim(flat_states, batch_size, beam_size)
      states = tf.gather_nd(states, topk_coordinates)
      cache = nest.map_structure(
                lambda t: _unmerge_beam_dim(t, batch_size, beam_size), flat_cache)
      cache = nest.map_structure(
                lambda t: tf.gather_nd(t, topk_coordinates), cache)
      topk_seq = tf.concat([topk_seq, tf.expand_dims(topk_ids, axis=2)], axis=2) # (batch, 2*beam, length+1)
      topk_finished = tf.equal(topk_ids, 3) # (batch, 2*beam)

      # 2. Extract the ones that have finished and haven't finished
      curr_scores = topk_scores + tf.to_float(topk_finished) * -inf # (batch, 2*beam)
      curr_scores = tf.reshape(curr_scores, [batch_size, 2, beam_size])
      _, topk_indexes = tf.nn.top_k(curr_scores, k=tf.cast(beam_size/2, tf.int32), sorted=True) ## [batch, 2, beam/2]
      topk_indexes_tmp = topk_indexes[:,1,:]+beam_size ##[batch, beam/2]
      topk_indexes = tf.concat([tf.expand_dims(topk_indexes[:,0,:],1), tf.expand_dims(topk_indexes_tmp,1)], axis=1)
      topk_indexes = tf.reshape(topk_indexes, [batch_size, beam_size])

      batch_pos_2 = compute_batch_indices(batch_size, beam_size)
      top_coordinates = tf.stack([batch_pos_2, topk_indexes], axis=2) # (batch, beam, 2) 
      alive_seq = tf.gather_nd(topk_seq, top_coordinates)
      alive_log_probs = tf.gather_nd(topk_log_probs, top_coordinates)
      alive_states = tf.gather_nd(states, top_coordinates)
      alive_cache_qkv = nest.map_structure(
                                           lambda t: tf.gather_nd(t, top_coordinates), cache)
    
      # 3. Recompute the contents of finished based on scores.

      finished_seq = tf.concat([finished_seq,tf.ones([batch_size, beam_size, 1], tf.int32)*3], axis=2)
      curr_scores = topk_scores + (1. - tf.to_float(topk_finished)) * -inf
      curr_finished_seq = tf.concat([finished_seq, topk_seq], axis=1)
      curr_finished_scores = tf.concat([finished_scores, curr_scores], axis=1)
      curr_finished_flags = tf.concat([finished_flags, topk_finished], axis=1)

      _, topk_indexes = tf.nn.top_k(curr_finished_scores, k=beam_size, sorted=True)
      top_coordinates = tf.stack([batch_pos_2, topk_indexes], axis=2)
      finished_seq = tf.gather_nd(curr_finished_seq, top_coordinates)
      finished_flags = tf.gather_nd(curr_finished_flags, top_coordinates)
      finished_scores = tf.gather_nd(curr_finished_scores, top_coordinates)
      
      return (flat_curr_scores, i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
              finished_flags, alive_states, alive_cache_qkv)

    def not_finished(flat_curr_scores, i, alive_seq, alive_log_probs, finished_seq, finished_scores, finished_flags, states, cache_qkv):
      return tf.logical_and(tf.logical_and(
        tf.reduce_any(tf.logical_not(finished_flags)),
        tf.less_equal(
          i, tf.reduce_min([tf.shape(encoder_output)[1] + 50,
                          self._config.test.max_target_length]))),tf.less_equal(i, tf.constant(100, dtype=tf.int32)))
      '''
      max_length_penalty = tf.pow(((5. + tf.to_float(decode_length)) / 6.), self._config.test.lp_alpha)
      lower_bound_alive_scores = alive_log_probs[:, 0] / max_length_penalty
      
      lowest_score_of_finished = tf.reduce_min(
              finished_scores * tf.to_float(finished_flags), axis=1)
      lowest_score_of_finished += (
              (1. - tf.to_float(tf.reduce_all(finished_flags, 1))) * -inf)
      bound_is_met = tf.reduce_all( # return True when lowest_score_of_finished > lower_bound_alive_scores
              tf.greater(lowest_score_of_finished, lower_bound_alive_scores))

      return tf.logical_and( # return True(do not finish) when i<decode_length and lowest_score_of_finished<lower_bound_alive_scores 
              tf.less(i, decode_length), tf.logical_not(bound_is_met))'''
    (flat_curr_scores, i, alive_seq, alive_log_probs, finished_seq,
     finished_scores, finished_flags, _, _)= tf.while_loop(
        cond=not_finished,
        body=step,
        loop_vars=[flat_curr_scores, 0, alive_seq, alive_log_probs, finished_seq,
         finished_scores, finished_flags, states, cache_qkv],
        shape_invariants=[
          tf.TensorShape([None, None, None]),
          tf.TensorShape([]),
          tf.TensorShape([None, None, None]),
          alive_log_probs.get_shape(),
          tf.TensorShape([None, None, None]),
          finished_scores.get_shape(),
          finished_flags.get_shape(),
          tf.TensorShape([None, None, None, None, None]),
          nest.map_structure(lambda t: get_state_shape_invariants(t), cache_qkv)],
        back_prop=False)

    alive_seq.set_shape((None, beam_size, None)) # (batch, beam, lenght)
    finished_seq.set_shape((None, beam_size, None))
    finished_seq = tf.where(
      tf.reduce_any(finished_flags, 1), finished_seq, alive_seq)
    finished_scores = tf.where(
      tf.reduce_any(finished_flags, 1), finished_scores, alive_log_probs)
    

    final_seq = finished_seq[:,0,:]
    return final_seq, finished_scores, alive_log_probs, finished_flags

  def test_output(self, decoder_output, reuse, decoder_scope):
    """During test, we only need the last prediction at each time."""
    with tf.variable_scope(decoder_scope, reuse=reuse):
      last_logits = layers.dense(decoder_output[:, -1], self._config.dst_vocab_size, use_bias=False,
                name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                reuse=True if self._config.tie_embedding_and_softmax else None)
    return last_logits

def _merge_beam_dim(tensor):
    """Reshapes first two dimensions in to single dimension.
        tensor: Tensor to reshape of shape [A, B, ...] --> [A*B, ...]
    """
    shape = shape_list(tensor)
    batch_size = shape[0]
    beam_size = shape[1]
    newshape = [batch_size*beam_size]+shape[2:]
    return tf.reshape(tensor, [batch_size*beam_size]+shape[2:])

def _unmerge_beam_dim(tensor, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].
        [batch_size*beam_size, ...] --> [batch_size, beam_size, ...]
    """
    shape = shape_list(tensor)
    new_shape = [batch_size]+[beam_size]+shape[1:]
    return tf.reshape(tensor, new_shape)

def _expand_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size.
        tensor: tensor to tile [batch_size, ...] --> [batch_size, beam_size, ...]
    """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size
    return tf.tile(tensor, tile_dims)
def compute_batch_indices(batch_size, beam_size):
    """Computes the i'th coodinate that contains the batch index for gathers.
    like [[0,0,0,0,],[1,1,1,1],..]
    """
    batch_pos = tf.range(batch_size * beam_size) // beam_size
    batch_pos = tf.reshape(batch_pos, [batch_size, beam_size])
    return batch_pos
def shape_list(x):
    if x.get_shape().dims is None:
        return tf.shape(x)
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i in xrange(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret
def get_state_shape_invariants(tensor):
    """Returns the shape of the tensor but sets middle dims to None."""
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)

