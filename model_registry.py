# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Convenience helpers for managing Params for datasets and models.
Typical usage will be to define and register a subclass of ModelParams
for each dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import tensorflow as tf

class _ModelRegistryHelper(object):
  # Global set of modules from which ModelParam subclasses have been registered.
  _REGISTERED_MODELS = {}

  @classmethod
  def _GetSourceInfo(cls, src_cls):
    """Gets a source info string given a source class."""
    return '%s:%d' % (inspect.getsourcefile(src_cls),
                      inspect.getsourcelines(src_cls)[-1])

  @classmethod
  def _RegisterModel(cls, src_cls):
    """Registers a ModelParams subclass in the global registry."""
    module = src_cls.__module__
    name = src_cls.__name__
    key = module + '.' + name
    model_info = cls._GetSourceInfo(src_cls)

    tf.logging.debug('Registering model %s', key)
    tf.logging.debug('%s : %s', (key, model_info))

    # Decorate param methods to add source info metadata.
    cls._REGISTERED_MODELS[key] = src_cls
    return key

  @classmethod
  def RegisterSingleTaskModel(cls, src_cls):
    """Class decorator that registers a `.SingleTaskModelParams` subclass."""
    cls._RegisterModel(src_cls)
    return src_cls

  @classmethod
  def RegisterMultiTaskModel(cls, src_cls):
    """Class decorator that registers a `.MultiTaskModelParams` subclass."""
    #cls._RegisterModel(src_cls)
    #return src_cls
    raise NotImplementedError()

  @staticmethod
  def GetAllRegisteredClasses():
    """Returns global registry map from model names to their param classes."""
    all_models = _ModelRegistryHelper._REGISTERED_MODELS
    if not all_models:
      tf.logging.warning('No classes registered.')
    return all_models

  @classmethod
  def GetClass(cls, class_key):
    """Returns a ModelParams subclass with the given `class_key`.
    Args:
      class_key: string key of the ModelParams subclass to return.
    Returns:
      A subclass of `~.base_model_params._BaseModelParams`.
    Raises:
      LookupError: If no class with the given key has been registered.
    """
    all_models = cls.GetAllRegisteredClasses()
    if class_key not in all_models:
      raise LookupError('Model %s not found. Known models:\n%s' %
                        (class_key, '\n'.join(sorted(all_models.keys()))))
    return all_models[class_key]


# pyformat: disable
# pylint: disable=invalid-name
RegisterSingleTaskModel = _ModelRegistryHelper.RegisterSingleTaskModel
RegisterMultiTaskModel = _ModelRegistryHelper.RegisterMultiTaskModel
GetAllRegisteredClasses = _ModelRegistryHelper.GetAllRegisteredClasses
GetClass = _ModelRegistryHelper.GetClass
# pylint: enable=invalid-name
# pyformat: enable