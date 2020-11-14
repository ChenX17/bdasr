# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

from core import layers
from core.layers import dense
from core import common_layers
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e5):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.
  
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).
  
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  
  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  channels = tf.shape(x)[2]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) /
    (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return x + signal

def bd_add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e5):
  """for bidirectional decoder
  Args:
    x: a Tensor with shape [2, batch, length, channels]
    min_timescale: a float
    max_timescale: a float
  
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[2]
  channels = tf.shape(x)[3]
  position = tf.to_float(tf.range(length))
  num_timescales = channels // 2
  log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) /
    (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, 1, length, channels])
  #signal = tf.concat([signal, signal],axis=0)
  return x + signal

def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  
  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase in one of the positional dimensions.
  
  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.
  
  The use of relative position is possible because sin(a+b) and cos(a+b) can be
  experessed in terms of b, sin(a) and cos(a).
  
  x is a Tensor with n "positional" dimensions, e.g. one dimension for a
  sequence or two dimensions for an image
  
  We use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels // (n * 2). For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.
  
  Args:
    x: a Tensor with shape [batch, d1 ... dn, channels]
    min_timescale: a float
    max_timescale: a float
  
  Returns:
    a Tensor the same shape as x.
  """
  static_shape = x.get_shape().as_list()
  num_dims = len(static_shape) - 2
  channels = tf.shape(x)[-1]
  num_timescales = channels // (num_dims * 2)
  log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) /
    (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
    tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  for dim in xrange(num_dims):
    length = tf.shape(x)[dim + 1]
    position = tf.to_float(tf.range(length))
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
      inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    prepad = dim * 2 * num_timescales
    postpad = channels - (dim + 1) * 2 * num_timescales
    signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
    for _ in xrange(1 + dim):
      signal = tf.expand_dims(signal, 0)
    for _ in xrange(num_dims - 1 - dim):
      signal = tf.expand_dims(signal, -2)
    x += signal
  return x


def add_positional_embedding_nd(x, max_length, name):
  """Add n-dimensional positional embedding.
  
  Adds embeddings to represent the positional dimensions of the tensor.
  The input tensor has n positional dimensions - i.e. 1 for text, 2 for images,
  3 for video, etc.
  
  Args:
    x: a Tensor with shape [batch, p1 ... pn, depth]
    max_length: an integer.  static maximum size of any dimension.
    name: a name for this layer.
  
  Returns:
    a Tensor the same shape as x.
  """
  static_shape = x.get_shape().as_list()
  dynamic_shape = tf.shape(x)
  num_dims = len(static_shape) - 2
  depth = static_shape[-1]
  base_shape = [1] * (num_dims + 1) + [depth]
  base_start = [0] * (num_dims + 2)
  base_size = [-1] + [1] * num_dims + [depth]
  for i in xrange(num_dims):
    shape = base_shape[:]
    start = base_start[:]
    size = base_size[:]
    shape[i + 1] = max_length
    size[i + 1] = dynamic_shape[i + 1]
    var = (tf.get_variable(
      name + "_%d" % i, shape,
      initializer=tf.random_normal_initializer(0, depth ** -0.5))
         * (depth ** 0.5))
    x += tf.slice(var, start, size)
  return x


def embedding_to_padding(emb):
  """Input embeddings -> is_padding.
  
  We have hacked symbol_modality to return all-zero embeddings for padding.
  
  Args:
    emb: a Tensor with shape [..., depth].
  Returns:
    a boolean Tensor with shape [...].
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.equal(emb_sum, 0.0)


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.
  
  Args:
   length: a Scalar.
  
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
  ret = -1e9 * (1.0 - lower_triangle)
  return tf.reshape(ret, [1, 1, length, length])

def attention_bias_higher_triangle(length):
  """Create an bias tensor to be added to attention logits for reverse training.
  
  Args:
   length: a Scalar.
  
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  lower_triangle = tf.matrix_band_part(tf.ones([length, length]), 0, -1)
  ret = -1e9 * (1.0 - lower_triangle)
  return tf.reshape(ret, [1, 1, length, length])

def bd_attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.
  
  Args:
   length: a Scalar.
  
  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
  ret = -1e9 * (1.0 - lower_triangle)
  ret = tf.reshape(ret, [1, 1, length, length])
  return ret

def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.
  
  Args:
    memory_padding: a boolean `Tensor` with shape [batch, memory_length].
  
  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = tf.to_float(memory_padding) * -1e9
  return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.
  
  The first of these two dimensions is n.
  
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  
  Returns:
    a Tensor with shape [..., n, m/n]
  """
  old_shape = x.get_shape().dims
  last = old_shape[-1]
  new_shape = old_shape[:-1] + [n] + [last // n if last else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
  ret.set_shape(new_shape)
  return ret


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.
  
  Args:
    x: a Tensor with shape [..., a, b]
  
  Returns:
    a Tensor with shape [..., ab]
  """
  old_shape = x.get_shape().dims
  a, b = old_shape[-2:]
  new_shape = old_shape[:-2] + [a * b if a and b else None]
  ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
  ret.set_shape(new_shape)
  return ret


def split_heads(x, num_heads):
  """Split channels (dimension 3) into multiple heads (becomes dimension 1).
  
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])

def sb_split_heads(x, num_heads):
  return tf.transpose(split_last_dimension(x, num_heads), [0, 1, 3, 2, 4])

def combine_heads(x):
  """Inverse of split_heads.
  
  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
  
  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def sb_combine_heads(x):
    return combine_last_two_dimensions(tf.transpose(x, [0, 1, 3, 2, 4]))


def attention_image_summary(attn, image_shapes=None):
  """Compute color image summary.
  
  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional quadruple of integer scalars.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
  """
  num_heads = attn.get_shape().as_list()[1]
  # [batch, query_length, memory_length, num_heads]
  image = tf.transpose(attn, [0, 2, 3, 1])
  image = tf.pow(image, 0.2)  # for high-dynamic-range
  # Each head will correspond to one of RGB.
  # pad the heads to be a multiple of 3
  image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, -num_heads % 3]])
  image = split_last_dimension(image, 3)
  image = tf.reduce_max(image, 4)
  if image_shapes is not None:
    q_rows, q_cols, m_rows, m_cols = list(image_shapes)
    image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
    image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
    image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
  tf.summary.image("attention", image, max_outputs=1)


def dot_product_attention(q,
              k,
              v,
              bias,
              dropout_rate=0.0,
              summaries=False,
              image_shapes=None,
              name=None):
  """dot-product attention.
  
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string
  
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

    #if summaries and not tf.get_variable_scope().reuse:
      #attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)

def latentgnn_attention(q,
              k,
              v,
              encoder_bias,
              decoder_bias,
              dropout_rate=0.0,
              summaries=False,
              image_shapes=None,
              latent_k_dim=10,
              latent_v_dim=10,
              name=None):
  """latent_gnn attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    # ----------------------------------
    # Generated projection matrix
    # ----------------------------------
    # generate projection matrix for q
    # (B, heads, length_q, laten_q_dim)
    q_proj_matrix = tf.layers.conv2d(
            q, latent_k_dim, (1,1),
            name="q_proj_generate"
        )
    # (B, heads, laten_q_dim, length_q)
    q_proj_matrix = tf.nn.l2_normalize(q_proj_matrix, axis=-1)
    # (B, heads, laten_q_dim, length_q)
    q_proj_matrix = tf.transpose(q_proj_matrix, perm=[0, 1, 3, 2])
    # (B, heads, length_k, laten_k_dim)
    k_proj_matrix = tf.layers.conv2d(
            k, latent_k_dim, (1,1),
            name="k_proj_generate"
        )
    # (B, heads, laten_k_dim, length_k)
    k_proj_matrix = tf.nn.l2_normalize(k_proj_matrix, axis=-1)
    # (B, heads, laten_k_dim, length_k)
    k_proj_matrix = tf.transpose(k_proj_matrix, perm=[0, 1, 3, 2])
    if decoder_bias is not None and encoder_bias is not None:
        q_proj_matrix = q_proj_matrix + encoder_bias
        k_proj_matrix = k_proj_matrix + encoder_bias
    if decoder_bias is None and encoder_bias is not None:
        k_proj_matrix = k_proj_matrix + encoder_bias
    #q_proj_matrix = tf.nn.softmax(q_proj_matrix, axis=-1)
    #k_proj_matrix = tf.nn.softmax(k_proj_matrix, axis=-1)
    v_proj_matrix = tf.layers.conv2d(
            v, latent_v_dim, (1,1),
            name="v_proj_generate"
        )
    v_proj_matrix = tf.nn.l2_normalize(v_proj_matrix, axis=-1)
    #v_proj_matrix = tf.nn.softmax(v_proj_matrix, axis=2)
    v_proj_matrix = tf.transpose(v_proj_matrix, perm=[0, 1, 3, 2])
    # ----------------------------------
    # Step-1: Project the feature into latent space
    # ----------------------------------
    # (B, heads, laten_q_dim, channel_q)
    latent_q_feat = tf.matmul(q_proj_matrix, q)
    latent_q_feat = tf.nn.l2_normalize(latent_q_feat, dim=-1)
    # (B, heads, laten_k_dim, channel_k) # channel_k = channel_q
    latent_k_feat = tf.matmul(k_proj_matrix, k)
    latent_k_feat = tf.nn.l2_normalize(latent_k_feat, dim=-1)
    # (B, heads, laten_v_dim, channel_v)
    latent_v_feat = tf.matmul(v_proj_matrix, v)
    # ----------------------------------
    # Step-2: Self-attention within latent space
    # ----------------------------------
    latent_attn_logits = tf.matmul(latent_q_feat, latent_k_feat, transpose_b=True)
    # (B, heads, latent_q_dim, latent_k_dim)
    latent_attn_score = tf.nn.softmax(latent_attn_logits, name='latent_attention_score')
    #(B, heads, length_q, laten_q_dim) * (B, heads, latent_q_dim, channel_v)
    #weights = tf.nn.dropout(latent_attn_score, 1.0 - dropout_rate)
    update_latent_v = tf.matmul(latent_attn_score, latent_v_feat)
    
    # add feature transformation
    channel_v = update_latent_v.get_shape().as_list()[3]
    update_latent_v = tf.nn.relu(
            tf.layers.conv2d(
                update_latent_v, channel_v, (1,1),
                name="latent_v_transformation"
                ))
    # ----------------------------------
    # Step-3: Re-project latent space into visible space
    # ----------------------------------
    q_reproj_matrix = tf.nn.relu(
            tf.layers.conv2d(
                q, latent_k_dim, (1,1),
                name="q_reproj_generate"
                ))
    q_reproj_matrix = tf.transpose(q_reproj_matrix, perm=[0, 1, 3, 2])
    # (B, heads, latent_q, channel_v)
    visible_out = tf.matmul(q_reproj_matrix, update_latent_v, transpose_a=True)
    
    channel_out = visible_out.get_shape().as_list()[3]
    visible_out = tf.nn.relu(
            tf.layers.conv2d(
                visible_out, channel_out, (1,1),
                name="visible_out_transformation"
                ))
    # Reshape
    if summaries and not tf.get_variable_scope().reuse:
      attention_image_summary(latent_attn_score, image_shapes)
    return visible_out

def dot_product_attention_with_atten_probs(q,
              k,
              v,
              bias,
              dropout_rate=0.0,
              summaries=False,
              image_shapes=None,
              name=None):
  """dot-product attention.
  
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string
  
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    if bias is not None:
      logits += bias
    atten_probs = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(atten_probs, 1.0 - dropout_rate)

    if summaries and not tf.get_variable_scope().reuse:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v), atten_probs

def dot_product_attention_sigmoid(q,
                  k,
                  v,
                  bias,
                  dropout_rate=0.0,
                  summaries=False,
                  image_shapes=None,
                  name=None):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]):
    # [batch, num_heads, query_length, memory_length]
    logits = tf.matmul(q, k, transpose_b=True)

    if bias is not None:
      logits += bias
    weights = tf.nn.sigmoid(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)

    if summaries and not tf.get_variable_scope().reuse:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def multihead_attention(query_antecedent,
            memory_antecedent,
            bias,
            total_key_depth,
            total_value_depth,
            output_depth,
            num_heads,
            dropout_rate,
            reserve_last=False,
            summaries=False,
            image_shapes=None,
            name=None):
  """Multihead scaled-dot-product attention with input/output transformations.
  
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
    name: an optional string
  
  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    if memory_antecedent is None:
      # self attention
      combined = dense(
        query_antecedent,
        total_key_depth * 2 + total_value_depth,
        name="qkv_transform")
      q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=2)
    else:
      q = dense(
        query_antecedent, total_key_depth, name="q_transform")
      combined = dense(
        memory_antecedent,
        total_key_depth + total_value_depth,
        name="kv_transform")
      k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
    if reserve_last:
      q = q[:, -1:, :]
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head ** -0.5
    x = dot_product_attention(
      q, k, v, bias, dropout_rate, summaries, image_shapes)
    x = combine_heads(x)
    x = dense(x, output_depth, name="output_transform")
    return x

def multihead_attention_with_latentgnn(query_antecedent,
                        memory_antecedent,
                        encoder_bias,
                        decoder_bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        reserve_last=False,
                        summaries=False,
                        image_shapes=None,
                        name=None,
                        latent_k_dim=50,
                        latent_v_dim=50):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    reserve_last: a boolean
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
  name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name,
      default_name="multihead_attention_with_latentgnn",
      values=[query_antecedent, memory_antecedent]):

    if memory_antecedent is None:
      # self attention
      combined = dense(query_antecedent, total_key_depth * 2 + total_value_depth, name="qkv_transform")
      q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=2)
    else:
      q = dense(query_antecedent, total_key_depth, name="q_transform")
      combined = dense(memory_antecedent,total_key_depth + total_value_depth, name="kv_transform")
      k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)
    if reserve_last:
      q = q[:, -1:, :]

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head ** -0.5
    x = latentgnn_attention(
      q, k, v, encoder_bias, decoder_bias, dropout_rate, summaries, image_shapes, latent_k_dim, latent_v_dim)
    x = combine_heads(x)
    x = x + query_antecedent
    x = dense(x, output_depth, name="output_transform")
    return x

def multihead_attention_with_atten_probs(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        reserve_last=False,
                        summaries=False,
                        image_shapes=None,
                        name=None):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    reserve_last: a boolean
    summaries: a boolean
    image_shapes: optional quadruple of integer scalars for image summary.
    If the query positions and memory positions represent the
    pixels of a flattened image, then pass in their dimensions:
      (query_rows, query_cols, memory_rows, memory_cols).
  name: an optional string

  Returns:
    A Tensor.
  """
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):

    if memory_antecedent is None:
      # self attention
      combined = dense(query_antecedent, total_key_depth * 2 + total_value_depth, name="qkv_transform")
      q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=2)
    else:
      q = dense(query_antecedent, total_key_depth, name="q_transform")
      combined = dense(memory_antecedent, total_key_depth + total_value_depth, name="kv_transform")
      k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

    if reserve_last:
      q = q[:, -1:, :]

    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head ** -0.5
    x, atten_probs = dot_product_attention_with_atten_probs(
      q, k, v, bias, dropout_rate, summaries, image_shapes)
    x = combine_heads(x)
    x = dense(x, output_depth, name="output_transform")
    return x, atten_probs


def sb_dot_product_attention_for_decoding(q,
                          k,
                          v,
                          bias,
                          batch_size=None,
                          beam_size=None,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        # dropping out the attention links for each of the heads
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        #if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights, image_shapes)
        final_l2r = tf.matmul(weights, v) ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]

        ## calculate final_r2l
        #shape = shape_list(k)
        # k = tf.Print(k, [k, k.shape,'test'], message="K is: ")
        # k=tf.Print(k,[k,k.shape,'test', k],message='Debug message:',summarize=2)
        #new_shape = [batch_size]+[2]+[tf.cast(beam_size/2,tf.int32)]+shape[1:]
        #k_ = tf.reshape(k, new_shape) ## [batch, 2, beam/2, num_heads, length_tmp, hidden_size/num_heads]
        #k_ = tf.reverse(k_,[1])
        #v_ = tf.reshape(v, new_shape)
        #v_ = tf.reverse(v_,[1])
        '''
        #for all l2r to attend first best r2l beam
        k_1 = k_[:, 0, 0,:, :, :]
        k_2 = k_[:, 1, 0,:, :, :]
        k_1 = tf.tile(tf.expand_dims(tf.expand_dims(k_1,1),1),[1, 1, tf.cast(beam_size/2, tf.int32), 1, 1, 1])
        k_2 = tf.tile(tf.expand_dims(tf.expand_dims(k_2,1),1),[1, 1, tf.cast(beam_size/2, tf.int32), 1, 1, 1])
        k_ = tf.concat([k_1,k_2], axis=1)
        v_1 = v_[:, 0, 0,:, :, :]
        v_2 = v_[:, 1, 0,:, :, :]
        v_1 = tf.tile(tf.expand_dims(tf.expand_dims(v_1,1),1),[1, 1, tf.cast(beam_size/2, tf.int32), 1, 1, 1])
        v_2 = tf.tile(tf.expand_dims(tf.expand_dims(v_2,1),1),[1, 1, tf.cast(beam_size/2, tf.int32), 1, 1, 1])
        v_ = tf.concat([v_1,v_2], axis=1)
        '''
        #shape_ = shape_list(k_)
        #new_shape_ = [batch_size*beam_size]+shape_[3:]
        #k_ = tf.reshape(k_, new_shape_) ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        #v_ = tf.reshape(v_, new_shape_)
        
        

        #logits_ = tf.matmul(q, k_, transpose_b=True)
        #logits_ += bias
        #weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        #weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        #final_r2l = tf.matmul(weights_, v_)

        #lamda = tf.get_variable('lamda', shape=[1], initializer = tf.ones_initializer)
        #final_all = final_l2r + lamda*tf.tanh(final_r2l)
        #final_all = final_l2r + 0*final_r2l ## [batch*beam, num_heads, length_tmp, hidden_size/num_heads]
        return final_l2r

def sb_dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          summaries=False,
                          image_shapes=None,
                          name=None):
    """dot-product attention.
        q: a Tensor with shape [batch, heads, length_q, depth_k]
    """
    with tf.variable_scope(
            name, default_name="sb_dot_product_attention", values=[q, k, v]):
        # [2, batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        bias = tf.expand_dims(bias, axis=0)
        logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights_l2r")
        weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
        #if summaries and not tf.get_variable_scope().reuse:
        #    attention_image_summary(weights[0], image_shapes)
        final_l2r = tf.matmul(weights, v) ## [2, batch, num_heads, length, hidden_size/num_heads]

        ## calculate final_r2l
        #k_ = tf.reverse(k, [0])
        #v_ = tf.reverse(v, [0])
        #logits_ = tf.matmul(q, k_, transpose_b=True)
        #logits_ += bias ### modify err, logits --> logits_
        #weights_ = tf.nn.softmax(logits_, name="attention_weights_r2l")
        #weights_ = tf.nn.dropout(weights_, 1.0 - dropout_rate)
        #final_r2l = tf.matmul(weights_, v_)
        #final_all = final_l2r + 0*tf.nn.dropout(final_r2l, 1)
        #final_all = final_l2r + 0.1*tf.tanh(tf.nn.dropout(final_r2l, 1-0.3))
        return final_l2r ## [2, batch, num_heads, length, hidden_size/num_heads]

def sb_multihead_attention(query_antecedent,
                           memory_antecedent,
                           bias,
                           total_key_depth,
                           total_value_depth,
                           output_depth,
                           num_heads,
                           dropout_rate,
                           cache=None,
                           summaries=False,
                           image_shapes=None,
                           name=None,
                           is_decoding=False):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """

    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
        # self attention
            combined = common_layers.sb_conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=3) ## 2-->3
        else:
            q = common_layers.sb_conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

            k = tf.concat([tf.expand_dims(k,0), tf.expand_dims(k,0)], axis=0) ## [2, batch, length, hidden_size]
            v = tf.concat([tf.expand_dims(v,0), tf.expand_dims(v,0)], axis=0)

        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)

        q = sb_split_heads(q, num_heads)
        k = sb_split_heads(k, num_heads)
        v = sb_split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None: ## decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention( ## for training
                q, k, v, bias, dropout_rate, summaries, image_shapes) ## q: [2, num_heads, length_tmp, lenght]
        else: ## enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = sb_combine_heads(x)
        x = common_layers.sb_conv1d(x, output_depth, 1, name="output_transform")
        return x


def sb_multihead_attention_for_decoding(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        batch_size=None,
                        beam_size=None,
                        cache=None,
                        summaries=False,
                        image_shapes=None,
                        name=None):
    """Multihead scaled-dot-product attention with input/output transformations.
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
    """

    with tf.variable_scope(
                name,
                default_name="sb_multihead_attention",
                values=[query_antecedent, memory_antecedent]):
        if memory_antecedent is None:
            # self attention
            combined = common_layers.conv1d(
                query_antecedent,
                total_key_depth * 2 + total_value_depth,
                1,
                name="qkv_transform")
            q, k, v = tf.split(
                combined, [total_key_depth, total_key_depth, total_value_depth], axis=2)
        else:
            q = common_layers.conv1d(
                query_antecedent, total_key_depth, 1, name="q_transform")
            combined = common_layers.conv1d(
                memory_antecedent,
                total_key_depth + total_value_depth,
                1,
                name="kv_transform")
            k, v = tf.split(combined, [total_key_depth, total_value_depth], axis=2)

        if cache is not None:
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                             "for details.")
            k = cache["k"] = tf.concat([cache["k"], k], axis=1)
            v = cache["v"] = tf.concat([cache["v"], v], axis=1)
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        key_depth_per_head = total_key_depth // num_heads
        q *= key_depth_per_head**-0.5
        if memory_antecedent is None: ## decoder self attention (synchronous bidirectional att)
            x = sb_dot_product_attention_for_decoding( ## for decoding
                q, k, v, bias, batch_size, beam_size, dropout_rate, summaries, image_shapes) ## q: [batch, num_heads, length_tmp, lenght]
        else: ## enc-dec attention
            x = dot_product_attention(
                q, k, v, bias, dropout_rate, summaries, image_shapes)
        x = combine_heads(x)
        x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
        return x
