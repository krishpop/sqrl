import wandb
import gin
import trainer
import os
import os.path as osp
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', './tfagents/safe-sac-sweeps/', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_str', 'MinitaurGoalVelocityEnv-v0', 'Environment string')
flags.DEFINE_integer('batch_size', 256, 'batch size used for training')
flags.DEFINE_float('safety_gamma', 0.7, 'Safety discount term used for TD backups')
flags.DEFINE_float('target_safety', 0.1, 'Target safety for safety critic')
flags.DEFINE_integer('target_entropy', -16, 'Target entropy for policy')
flags.DEFINE_float('lr', 3e-4, 'Learning rate for all optimizers')
flags.DEFINE_float('reward_scale_factor', 1.0, 'Reward scale factor for SacAgent')
flags.DEFINE_multi_string('gin_files', ['minitaur_default.gin', 'sac_safe_online.gin', 'networks.gin'],
                          'gin files to load')
flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
flags.DEFINE_integer('seed', None, 'Seed to seed envs and algorithm with')

wandb.init(sync_tensorboard=True, entity='krshna', project='safemrl', config=FLAGS)


def gin_bindings_from_config(config):
  gin_bindings = []
  gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.safety_gamma = {}'.format(config.safety_gamma))
  gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.target_safety = {}'.format(config.target_safety))
  gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.target_entropy = {}'.format(config.target_entropy))
  gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.reward_scale_factor = {}'.format(config.reward_scale_factor))
  gin_bindings.append('LEARNING_RATE = {}'.format(config.lr))
  gin_bindings.append('ENV_STR = "{}"'.format(config.env_str))
  return gin_bindings


def main(_):
  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))
  config = wandb.config
  config.root_dir = osp.join(config.root_dir, str(os.environ.get('WANDB_RUN_ID', 0)))
  gin_files = config.gin_files
  gin_bindings = gin_bindings_from_config(config)
  gin.parse_config_files_and_bindings(gin_files, gin_bindings)
  tf.config.threading.set_inter_op_parallelism_threads(12)
  trainer.train_eval(FLAGS.root_dir, batch_size=config.batch_size, seed=FLAGS.seed,
                     eager_debug=FLAGS.eager_debug)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
