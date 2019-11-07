import gin
import os
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from algos.train_eval import train_eval
from comet_ml import Experiment


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None, 'Path to the trainer config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')
flags.DEFINE_boolean('run_eval', False, 'Whether or not to run eval')

FLAGS = flags.FLAGS


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  gin.finalize()
  experiment = Experiment(api_key="ZIIxUqFtxJ6uSt34ifrIAcZVw",
                          project_name="safemrl", workspace="krishpop")
  train_eval(FLAGS.root_dir, run_eval=FLAGS.run_eval)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
