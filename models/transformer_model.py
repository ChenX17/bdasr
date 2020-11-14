# coding=utf-8
import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops

import model_registry
from core import layers
from core import layers_with_attention
from core.layers import embedding, residual, dense, ff_hidden
from core.layers_with_attention import multihead_attention
from core.utils import average_gradients, shift_right
from core.utils import learning_rate_decay
from models.base_model import BaseModel

@model_registry.RegisterSingleTaskModel
class TransformerModel(BaseModel):
  """Model for transformer architecture."""

  def __init__(self, *args, **kargs):
    super(TransformerModel, self).__init__(*args, **kargs)
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
                   vocab_size=self._config.vocab_size,
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
                   vocab_size=self._config.vocab_size,
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
    atten_probs_list = []
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
        multihead_out, atten_probs = layers_with_attention.multihead_attention_with_atten_probs(
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
                        summaries=True)
        atten_probs_list.append(atten_probs)
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

    return decoder_output, new_cache, atten_probs_list[-1]
