# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to average values of variables in a list of checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
            tf.gfile.Exists(path + ".index"))


def avg_model(checkpoints, checkpoints_dir):
    from ctypes import cdll

    # cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so.6')
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "11"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    avg_model_prefix = os.path.join('/datadisk/model/aishell1/avg_model',checkpoints_dir.split('/')[-1])
    output_path = os.path.join(avg_model_prefix,'averaged.ckpt')

    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")
    if checkpoints_dir:
        checkpoints = [os.path.join(checkpoints_dir, c) for c in checkpoints]
    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
    if not checkpoints:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % checkpoints)

    # Read variables from all checkpoints and average them.
    tf.logging.info("Reading variables and averaging checkpoints:")
    for c in checkpoints:
        tf.logging.info("%s ", c)
    var_list = tf.contrib.framework.list_variables(checkpoints[0])
    var_values, var_dtypes = {}, {}
    for (name, shape) in var_list:
        if not name.startswith("global_step"):
            var_values[name] = np.zeros(shape)
    for checkpoint in checkpoints:
        reader = tf.contrib.framework.load_checkpoint(checkpoint)
        for name in var_values:
            #import pdb;pdb.set_trace()
            if 'alpha' in name:
                if 'Adam' not in name:
                    #import pdb;pdb.set_trace()
                    print(name + '=' + str(reader.get_tensor(name)))             
            tensor = reader.get_tensor(name)
            var_dtypes[name] = tensor.dtype
            var_values[name] += tensor
            if name == 'douyin_4000h/uls/atten/atten_step_counter/var':
                var_values[name] = tensor
                var_dtypes[name] = tensor.dtype
                #print(tensor.dtype)
                #print(var_values[name].dtype)
        tf.logging.info("Read from checkpoint %s", checkpoint)
    #name2w = []
    for name in var_values:  # Average.
        #name2w.append(name + '\n')
        if name != 'douyin_4000h/uls/atten/atten_step_counter/var':
            var_values[name] /= len(checkpoints)
        else:
            if 'alpha' in name:
                print(var_values[name])
    #f = open('/datadisk/projects/transformerasr/transformer/third_party/tensor2tensor/paraname.txt','w')
    #f.writelines(name2w)
    #f.close
    tf_vars = [
        tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
        for v in var_values
    ]
    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    global_step = tf.Variable(
        0, name="global_step", trainable=False, dtype=tf.int64)
    saver = tf.train.Saver(tf.all_variables())

    # Build a model consisting only of variables, set them to the average values.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for p, assign_op, (name, value) in zip(placeholders, assign_ops,
                                               six.iteritems(var_values)):
            sess.run(assign_op, {p: value})
        # Use the built saver to save the averaged checkpoint.
        saver.save(sess, output_path, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", output_path)
    print("Averaged checkpoints saved in %s"%output_path)


if __name__ == "__main__":
    tf.app.run()
