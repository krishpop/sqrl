# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

r"""General TF-Agents trainer executable.

Runs training on a TFAgent in a specified environment. It is recommended that
the agent be configured using Gin-config and the --gin_file flag, but you
can also import the train function and pass an agent class that you have
configured manually.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import os.path as osp

from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import gin
import tensorflow as tf
import random
import numpy as np
tf.compat.v1.enable_v2_behavior()

from safemrl import trainer
from safemrl import train_sc

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('load_dir', None,
                    'loading directory for loading checkpoint')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean('train_sc', False, 'loads checkpointed buffer and '
                                             'trains safety critic offline')
flags.DEFINE_integer('num_steps', 20000, 'number of training steps')
flags.DEFINE_integer('batch_size', 256, 'batch size per train step')
flags.DEFINE_float('lr', None, 'safety critic optimizer learning rate')
flags.DEFINE_float('sc_bias_init_val', None, 'value for safety critic '
                                            'constant bias initializer')
flags.DEFINE_float('sc_kernel_scale', None, 'value for safety critic '
                                            'kernel variance scaling')
flags.DEFINE_float('fail_weight', None, 'how much to weight failure experience')
flags.DEFINE_boolean('finetune', False, 'whether or not to finetune')
flags.DEFINE_boolean('monitor', False, 'whether or not to use monitoring')
flags.DEFINE_boolean('debug', False, 'set log level to debug if True')
flags.DEFINE_boolean('debug_summaries', False, 'log debug summaries to tensorboard')
flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
flags.DEFINE_integer('seed', None, 'Seed to seed envs and algorithm with')

FLAGS = flags.FLAGS


def gin_to_config(config_str):
  bindings = [b for b in config_str.split('\n') if ' = ' in b]
  config_dict = {}
  for b in bindings:
    key, formatted_val = b.split(' = ')
    config_dict[key] = gin.config.query_parameter(key)
  return config_dict


def main(_):
  logging.set_verbosity(logging.INFO)
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
  logging.debug('Executing eagerly: %s', tf.executing_eagerly())

  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))

  root_dir = FLAGS.root_dir or FLAGS.load_dir
  if os.environ.get('EXP_DIR') and not os.path.exists(root_dir):
    root_dir = os.path.join(os.environ.get('EXP_DIR'), root_dir)

  gin_files = FLAGS.gin_file or []
  if FLAGS.train_sc:
    op_config = osp.join(root_dir, 'train/operative_config-0.gin')
    if osp.exists(op_config):
      gin_files.append(op_config)
  logging.debug('parsing config files: %s', gin_files)

  gin_bindings = FLAGS.gin_param or []
  if FLAGS.num_steps:
    gin_bindings.append('NUM_STEPS = {}'.format(FLAGS.num_steps))
  if FLAGS.lr:
    gin_bindings.append('SC_LEARNING_RATE = {}'.format(FLAGS.lr))
  logging.debug('parsing gin bindings: %s', gin_bindings)
  gin.parse_config_files_and_bindings(gin_files, gin_bindings,
                                      skip_unknown=True)

  if FLAGS.seed:
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    logging.debug('Set seed: %d', FLAGS.seed)

  if FLAGS.train_sc:
    train_sc.train_eval(root_dir,
                        safety_critic_bias_init_val=FLAGS.sc_bias_init_val,
                        safety_critic_kernel_scale=FLAGS.sc_kernel_scale,
                        fail_weight=FLAGS.fail_weight,
                        seed=FLAGS.seed, monitor=FLAGS.monitor,
                        debug_summaries=FLAGS.debug_summaries)
  else:
    trainer.train_eval(root_dir, load_root_dir=FLAGS.load_dir,
                       pretraining=(not FLAGS.finetune), monitor=FLAGS.monitor,
                       eager_debug=FLAGS.eager_debug, seed=FLAGS.seed,
                       debug_summaries=FLAGS.debug_summaries)


if __name__ == '__main__':
  app.run(main)
