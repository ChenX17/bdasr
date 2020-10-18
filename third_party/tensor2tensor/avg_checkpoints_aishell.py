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

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoints", "",
                    "Comma-separated list of checkpoints to average.")
flags.DEFINE_string("prefix", "",
                    "Prefix (e.g., directory) to append to each checkpoint.")
flags.DEFINE_string("output_path", "/tmp/averaged.ckpt",
                    "Path to output the averaged checkpoint to.")


def checkpoint_exists(path):
    return (tf.gfile.Exists(path) or tf.gfile.Exists(path + ".meta") or
            tf.gfile.Exists(path + ".index"))


def main(_):
    from ctypes import cdll

    # cdll.LoadLibrary('/usr/local/cuda/lib64/libcudnn.so.6')
    import os

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

    #FLAGS.checkpoints = 'model_step_191000,model_step_192000,model_step_193000,model_step_194000,model_step_195000,model_step_196000,model_step_197000,model_step_198000,model_step_199000,model_step_200000'
    #FLAGS.checkpoints = 'model_step_341000,model_step_342000,model_step_343000,model_step_344000,model_step_345000,model_step_346000,model_step_347000,model_step_348000,model_step_349000,model_step_350000'
    FLAGS.checkpoints = 'model_step_61505,model_step_62711,model_step_55475,model_step_69947,model_step_73565'
    #FLAGS.output_path = '/datadisk/model/aishell2/avg_model/transformer_8_4_4gpus_158000_160000/averaged.ckpt'
    #FLAGS.checkpoints = 'model_epoch_36,model_epoch_39,model_epoch_35,model_epoch_40,model_epoch_38'
    #FLAGS.checkpoints = 'model_epoch_48,model_epoch_49,model_epoch_50,model_epoch_51,model_epoch_57'
    #FLAGS.checkpoints = 'model_epoch_48,model_epoch_49,model_epoch_50,model_epoch_46,model_epoch_43'
    #FLAGS.checkpoints = 'model_epoch_43,model_epoch_48,model_epoch_46,model_epoch_52,model_epoch_59'
    #FLAGS.checkpoints = 'model_epoch_63,model_epoch_58,model_epoch_61,model_epoch_56,model_epoch_62'
    FLAGS.checkpoints = 'model_epoch_67,model_epoch_56,model_epoch_66,model_epoch_45,model_epoch_55'
    FLAGS.checkpoints = 'model_epoch_43,model_epoch_31,model_epoch_38,model_epoch_32,model_epoch_40'
    #FLAGS.checkpoints = 'model_epoch_54,model_epoch_56,model_epoch_67,model_epoch_69,model_epoch_60'
    #FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_exp1/'
    #FLAGS.prefix = '/datadisk/model/aishell1/transformer_6_6_4gpus_1024units_exp3/'
    FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_6_4gpus_exp1/'
    FLAGS.prefix = '/datadisk/model/aishell1/transformer_8_4_4gpus_512units/'
    FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_512units_exp28/'
    FLAGS.prefix = '/datadisk/model/aishell1/reverse_transformer_8_4_4gpus_512units_exp3/'
    FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_6_6_4gpus_exp1/'
    FLAGS.prefix = '/datadisk/model/aishell2/transformer_8_4_4gpus_512units/'
    FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_512units_exp6/'
    FLAGS.prefix = '/datadisk/model/aishell1/transformer_8_4_4gpus_512units_exp5/'
    FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_512units_specaug_sp_exp4/'
    FLAGS.prefix = '/datadisk/model/aishell1/transformer_8_4_4gpus_512units_specaug_sp_exp1/'
    #FLAGS.prefix = '/datadisk/model/aishell2/bd/bd_transformer_8_4_4gpus_exp1/'
    #FLAGS.prefix = '/datadisk/model/aishell1/transformer_8_4_4gpus_512units_exp3/'
    #FLAGS.prefix = '/datadisk/model/aishell1/bd/bd_transformer_8_4_4gpus_1024unit_exp5/'
    FLAGS.output_path = '/datadisk/model/aishell1/avg_model/transformer_8_4_4gpus_512units_specaug_sp_exp1_best5/averaged.ckpt'

    # Get the checkpoints list from flags and run some basic checks.
    checkpoints = [c.strip() for c in FLAGS.checkpoints.split(",")]
    checkpoints = [c for c in checkpoints if c]
    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")
    if FLAGS.prefix:
        checkpoints = [FLAGS.prefix + c for c in checkpoints]
    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]
    #import pdb;pdb.set_trace()
    if not checkpoints:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % FLAGS.checkpoints)

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
        saver.save(sess, FLAGS.output_path, global_step=global_step)

    tf.logging.info("Averaged checkpoints saved in %s", FLAGS.output_path)


if __name__ == "__main__":
    tf.app.run()
