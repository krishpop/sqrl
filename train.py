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

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
import random
import numpy as np
tf.compat.v1.enable_v2_behavior()

from safemrl import trainer

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean('debug', False, 'set log level to debug if True')
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
  root_dir = FLAGS.root_dir
  if os.environ.get('EXP_DIR'):
    root_dir = os.path.join(os.environ.get('EXP_DIR'), root_dir)

  logging.debug('parsing config files: %s', FLAGS.gin_file)
  if FLAGS.seed:
    # bindings.append(('trainer.train_eval.seed', FLAGS.seed))
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)
    logging.debug('Set seed: %d', FLAGS.seed)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)

  trainer.train_eval(root_dir, eager_debug=FLAGS.eager_debug, seed=FLAGS.seed)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
