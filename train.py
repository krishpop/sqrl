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
tf.compat.v1.enable_v2_behavior()
import wandb

from safemrl import trainer

flags.DEFINE_string('root_dir', None,
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean('debug', False, 'set log level to debug if True')
flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
flags.DEFINE_boolean('wandb', False, 'Whether or not to log experiment to wandb')

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
  logging.info('Executing eagerly: %s', tf.executing_eagerly())
  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))
  logging.info('parsing config files: %s', FLAGS.gin_file)
  gin.parse_config_files_and_bindings(
      FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
  metrics_callback = None
  if FLAGS.wandb:
    wandb.init(sync_tensorboard=True, entity='krshna', project='safemrl')
    global stop_training
    stop_training = False
    early_stopping_fn = lambda: stop_training[0]
    @gin.configurable
    def metrics_callback(results, step, metric_name="Metrics/AverageReturn"):
      global stop_training
      metric_val = results[metric_name]
      stop_training = True
  else:
    early_stopping_fn = lambda: False
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.DEBUG)

  trainer.train_eval(FLAGS.root_dir, eval_metrics_callback=metrics_callback,
                     early_termination_fn=early_stopping_fn, debug_summaries=FLAGS.debug)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
