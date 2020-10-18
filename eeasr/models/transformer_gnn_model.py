# coding=utf-8
import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops

from eeasr import model_registry
from eeasr.core import layers
from eeasr.core import layers_with_attention
from eeasr.core.layers import embedding, residual, residual_gnn, dense, ff_hidden
from eeasr.core.layers_with_attention import multihead_attention_with_latentgnn
from eeasr.core.layers_with_attention import multihead_attention
from eeasr.core.utils import average_gradients, shift_right
from eeasr.core.utils import learning_rate_decay
from eeasr.models.base_model import BaseModel
from eeasr.core import utils
@model_registry.RegisterSingleTaskModel
class LatentGNNTransformerModel(BaseModel):
  """Model for transformer architecture."""

  def __init__(self, *args, **kargs):
    super(LatentGNNTransformerModel, self).__init__(*args, **kargs)
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
        encoder_output = residual_gnn(encoder_output,
                      multihead_attention_with_latentgnn(
                        query_antecedent=encoder_output,
                        memory_antecedent=None,
                        encoder_bias=layers_with_attention.attention_bias_ignore_padding(encoder_padding),
                        decoder_bias=layers_with_attention.attention_bias_ignore_padding(encoder_padding),
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        name='encoder_self_attention',
                        summaries=True,
                        latent_k_dim=self._config.latent_k_dim,
                        latent_v_dim=self._config.latent_v_dim),
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
    import pdb; pdb.set_trace()
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
    #atten_probs_list = []
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
                        output_depth=self._config.hidden_units,
                        name="decoder_self_attention",
                        reserve_last=True,
                        summaries=True),
                      dropout_rate=residual_dropout_rate)

        # Multihead Attention (vanilla attention)
        multihead_out = layers_with_attention.multihead_attention(
                        query_antecedent=decoder_output,
                        memory_antecedent=encoder_output,
                        bias=encoder_attention_bias,
                        total_key_depth=self._config.hidden_units,
                        total_value_depth=self._config.hidden_units,
                        output_depth=self._config.hidden_units,
                        num_heads=self._config.num_heads,
                        dropout_rate=attention_dropout_rate,
                        name="decoder_vanilla_attention",
                        reserve_last=True,
                        summaries=True)
        #atten_probs_list.append(atten_probs)
        decoder_output = residual(decoder_output, multihead_out,
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

  def build_train_model(self, test=True, reuse=None):
    """Build model for training. """
    logging.info('Build train model.')
    self.prepare_training()

    with self.graph.as_default():
      acc_list = []
      loss_list = []
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
            decoder_output = self.decoder(utils.shift_right(Y), encoder_output,
                                          is_training=True,
                                          reuse=i > 0 or None,
                                          decoder_scope=self.decoder_scope)
            acc, loss = self.train_output(decoder_output, Y,
                                             reuse=i > 0 or None,
                                             decoder_scope=self.decoder_scope)
            var_list = tf.trainable_variables()
            if self._config.train.var_filter:
              var_list = [v for v in var_list if
                  re.match(self._config.train.var_filter, v.name)]
            #l2_loss_list = []
            #for i in var_list:
              #l2_loss_list.append(self.l2_weight * tf.nn.l2_loss(i))
            #l2_loss = tf.reduce_sum(l2_loss_list)
            #loss += l2_loss
            acc_list.append(acc)
            loss_list.append(loss)

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
      #tf.summary.scalar('l2_loss', l2_loss)
      tf.summary.scalar('learning_rate', self.learning_rate)
      tf.summary.scalar('grads_norm', self.grads_norm)
      tf.summary.scalar('avg_abs_grads', avg_abs_grads)
      self.summary_op = tf.summary.merge_all()

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=100)

    # We may want to test the model during training.
    if test:
      self.build_test_model(reuse=True)
  def build_test_model(self, reuse=None):
    """Build model for inference."""
    logging.info('Build test model.')
    with self.graph.as_default(), tf.variable_scope(
        tf.get_variable_scope(), reuse=reuse):
      prediction_list = []
      loss_sum = 0
      for i, (X, Y, device) in enumerate(
          zip(self.src_pls, self.label_pls, self._devices)):
        with tf.device(device):
          logging.info('Build model on %s.' % device)
          dec_input = utils.shift_right(Y)

          # Avoid errors caused by empty input by a condition phrase.
          def true_fn():
            enc_output = self.encoder(X, is_training=False,
                reuse=i > 0 or None, encoder_scope=self.encoder_scope)
            prediction = self.beam_search(enc_output, reuse=i > 0 or None)
            dec_output = self.decoder(dec_input, enc_output, is_training=False,
                reuse=True, decoder_scope=self.decoder_scope)
            loss = self.test_loss(dec_output, Y, reuse=True,
                decoder_scope=self.decoder_scope)
            return prediction, loss

          def false_fn():
            return tf.zeros([0, 0], dtype=tf.int32), 0.0

          prediction, loss = tf.cond(tf.greater(tf.shape(X)[0], 0),
                                     true_fn, false_fn)
          loss_sum += loss
          prediction_list.append(prediction)

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
      #self.prediction = prediction_list
      self.prediction = tf.concat(prediction_list, axis=0)
      #self.atten_probs = atten_probs_list
      self.loss_sum = loss_sum

      self.saver = tf.train.Saver(var_list=tf.global_variables(),
          max_to_keep=100)
  def beam_search(self, encoder_output, reuse):
    """Beam search in graph."""
    beam_size = self._config.test.beam_size
    batch_size = tf.shape(encoder_output)[0]
    inf = 1e10

    def get_bias_scores(scores, bias):
      """If a sequence is finished, we only allow one alive branch.

      This function aims to give one branch a zero score and the rest
      -inf score.
      Args:
        scores: A real value array with shape [batch_size * beam_size,
          beam_size].
        bias: A bool array with shape [batch_size * beam_size].

      Returns:
        A real value array with shape [batch_size * beam_size, beam_size].
      """
      bias = tf.to_float(bias)
      b = tf.constant([0.0] + [-inf] * (beam_size - 1))
      b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
      return scores * (1 - bias[:, None]) + b * bias[:, None]

    def get_bias_preds(preds, bias):
      """
      If a sequence is finished, all of its branch should be </S> (3).
      Args:
        preds: A int array with shape [batch_size * beam_size, beam_size].
        bias: A bool array with shape [batch_size * beam_size].

      Returns:
        A int array with shape [batch_size * beam_size].
      """
      bias = tf.to_int32(bias)
      return preds * (1 - bias[:, None]) + bias[:, None] * 3

    # Prepare beam search inputs.
    # [batch_size, 1, *, hidden_units]
    encoder_output = encoder_output[:, None, :, :]
    # [batch_size, beam_size, feat_len, hidden_units]
    encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
    # [batch_size * beam_size, feat_len, hidden_units]
    encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1,
        encoder_output.get_shape()[-1].value])
    # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
    preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
    feat_len = tf.shape(encoder_output)[1]
    #atten_probs = tf.zeros([batch_size * beam_size, 0, feat_len],
                           #dtype=tf.float32)
    # [beam_size]
    scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)
    scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
    bias = tf.zeros_like(scores, dtype=tf.bool)  # 是否结束的标识位
    # 缓存的历史结果，[batch_size * beam_size, 0, num_blocks , hidden_units ]
    cache = tf.zeros([batch_size * beam_size, 0,
                      self._config.decoder_num_blocks,
                      self._config.hidden_units])

    def step(i, bias, preds, scores, cache):
      # Where are we.
      i += 1

      # Call decoder and get predictions.
      decoder_output, cache = self.decoder_with_caching(
          preds, cache, encoder_output, is_training=False,
          reuse=reuse, decoder_scope='decoder')
      last_preds, last_k_preds, last_k_scores = self.test_output(
          decoder_output, reuse=reuse, decoder_scope='decoder')
      last_k_preds = get_bias_preds(last_k_preds, bias)
      last_k_scores = get_bias_scores(last_k_scores, bias)
      # input t_atten_probs: [batch_size * beam_size, num_head, target_len, feat_len]
      # Only last step need.  [batch_size * beam_size, num_head, feat_len]
      #t_atten_probs = t_atten_probs[:,:,-1,:]
      # [batch_size * beam_size, feat_len]
      #t_atten_probs = tf.reduce_mean(t_atten_probs, axis=1)
      #t_atten_probs = t_atten_probs[:,0,:]  # fist head.
      # Update scores.
      # [batch_size * beam_size, beam_size]
      scores = scores[:, None] + last_k_scores
      # [batch_size, beam_size * beam_size]
      scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])

      # Pruning.
      scores, k_indices = tf.nn.top_k(scores, k=beam_size)
      scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
      base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None],
          multiples=[1, beam_size]), shape=[-1])
      base_indices *= beam_size ** 2
      # [batch_size * beam_size]
      k_indices = base_indices + tf.reshape(k_indices, shape=[-1])

      # Update predictions.
      last_k_preds = tf.gather(tf.reshape(last_k_preds, shape=[-1]),
                               indices=k_indices)
      preds = tf.gather(preds, indices=k_indices / beam_size)
      cache = tf.gather(cache, indices=k_indices / beam_size)
      #attent_probs = tf.gather(atten_probs, indices=k_indices / beam_size)
      # [batch_size * beam_size, i]
      preds = tf.concat((preds, last_k_preds[:, None]), axis=1)
      #atten_probs = tf.concat((atten_probs, t_atten_probs[:, None, :]), axis=1)

      # Whether sequences finished.
      bias = tf.equal(preds[:, -1], 3)  # </S>?

      return i, bias, preds, scores, cache

    def not_finished(i, bias, preds, scores, cache):
      return tf.logical_and(
        tf.reduce_any(tf.logical_not(bias)),
        tf.less_equal(
          i, tf.reduce_min([tf.shape(encoder_output)[1] + 50,
                           self._config.test.max_target_length])))

    i, bias, preds, scores, cache = tf.while_loop(
        cond=not_finished,
        body=step,
        loop_vars=[0, bias, preds, scores, cache],
        shape_invariants=[
          tf.TensorShape([]),
          tf.TensorShape([None]),
          tf.TensorShape([None, None]),
          tf.TensorShape([None]),
          tf.TensorShape([None, None, None, None])],
        back_prop=False)

    scores = tf.reshape(scores, shape=[batch_size, beam_size])
    preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
    lengths = tf.reduce_sum(tf.to_float(tf.not_equal(preds, 3)), axis=-1)  # [batch_size, beam_size]
    lp = tf.pow((5 + lengths) / (5 + 1), self._config.test.lp_alpha)  # Length penalty
    scores /= lp  # following GNMT
    max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
    max_indices += tf.range(batch_size) * beam_size
    preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

    final_preds = tf.gather(preds, indices=max_indices)
    #final_atten_probs = tf.gather(atten_probs, indices=max_indices)
    final_preds = final_preds[:, 1:]  # remove <S> flag
    return final_preds

  def encoder(self, encoder_input, is_training, reuse, encoder_scope):
    """Encoder."""
    with tf.variable_scope(encoder_scope, reuse=reuse):
      return self.encoder_impl(encoder_input, is_training)

  def decoder(self, decoder_input, encoder_output,
      is_training, reuse, decoder_scope):
    """Decoder"""
    with tf.variable_scope(decoder_scope, reuse=reuse):
      return self.decoder_impl(decoder_input, encoder_output, is_training)

  def decoder_with_caching(self, decoder_input, decoder_cache,
      encoder_output, is_training, reuse, decoder_scope):
    """Incremental Decoder"""
    with tf.variable_scope(decoder_scope, reuse=reuse):
      return self.decoder_with_caching_impl(
          decoder_input, decoder_cache, encoder_output, is_training)
