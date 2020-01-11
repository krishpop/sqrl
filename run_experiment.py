import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import wandb
import gin
import trainer
import os
import os.path as osp
import numpy as np

from absl import flags
from absl import app


FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', 'safe-sac-sweeps/cube_rotate/', 'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('env_str', 'SafemrlCube-v0', 'Environment string')
flags.DEFINE_integer('num_steps', 2000000, 'Number of training steps')
flags.DEFINE_integer('layer_size', 256, 'Number of training steps')
flags.DEFINE_integer('batch_size', 256, 'batch size used for training')
flags.DEFINE_float('safety_gamma', 0.7, 'Safety discount term used for TD backups')
flags.DEFINE_float('target_safety', 0.1, 'Target safety for safety critic')
flags.DEFINE_integer('target_entropy', None, 'Target entropy for policy')
flags.DEFINE_float('lr', None, 'Learning rate for all optimizers')
flags.DEFINE_float('actor_lr', 3e-4, 'Learning rate for actor')
flags.DEFINE_float('critic_lr', 3e-4, 'Learning rate for critic')
flags.DEFINE_float('entropy_lr', 3e-4, 'Learning rate for alpha')
flags.DEFINE_float('target_update_tau', 0.001, 'Factor for soft update of the target networks')
flags.DEFINE_integer('target_update_period', 1, 'Period for soft update of the target networks')
flags.DEFINE_integer('initial_collect_steps', 10000, 'Number of steps to collect with random policy')
flags.DEFINE_float('initial_log_alpha', 0., 'Initial value for log_alpha')
flags.DEFINE_float('gamma', 0.99, 'Future reward discount factor')
flags.DEFINE_float('reward_scale_factor', 1.0, 'Reward scale factor for SacAgent')
flags.DEFINE_multi_string('gin_files', ['cube_default.gin', 'sac_safe_online.gin'],
                          'gin files to load')
flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
flags.DEFINE_integer('seed', None, 'Seed to seed envs and algorithm with')

wandb.init(sync_tensorboard=True, entity='krshna', project='safemrl', config=FLAGS)


def gin_bindings_from_config(config):
  gin_bindings = []
  if 'sac_safe_online.gin' in config.gin_files:
    gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.safety_gamma = {}'.format(config.safety_gamma))
    gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.target_safety = {}'.format(config.target_safety))
    agent_prefix = 'safe_sac_agent.SafeSacAgentOnline'
  elif 'sac.gin' in config.gin_files:
    agent_prefix = 'sac_agent.SacAgent'
  gin_bindings.append(
      '{}.target_entropy = {}'.format(agent_prefix, config.target_entropy))
  gin_bindings.append(
      '{}.reward_scale_factor = {}'.format(agent_prefix, 10**config.reward_scale_factor))
  gin_bindings.append(
      '{}.target_update_tau = {}'.format(agent_prefix, config.target_update_tau))
  gin_bindings.append(
    '{}.target_update_period = {}'.format(agent_prefix, config.target_update_period))
  gin_bindings.append('{}.gamma = {}'.format(agent_prefix, config.gamma))
  if config.lr and not wandb.run.resumed:  # do not update config if resuming run
    if config.lr < 0:  # HACK: hp.loguniform not working
      config.update(dict(lr=10 ** config.lr), allow_val_change=True)
    gin_bindings.append('LEARNING_RATE = {}'.format(config.lr))
  elif not wandb.run.resumed:
    if config.actor_lr < 0:
      config.update(dict(actor_lr=10**config.actor_lr), allow_val_change=True)
    if config.critic_lr < 0:
      config.update(dict(critic_lr=10**config.critic_lr), allow_val_change=True)
    if config.entropy_lr < 0:
      config.update(dict(entropy_lr=10**config.entropy_lr), allow_val_change=True)
    gin_bindings.append('ac_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.actor_lr))
    gin_bindings.append('cr_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.critic_lr))
    gin_bindings.append('al_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.entropy_lr))

  gin_bindings.append("INITIAL_NUM_STEPS = {}".format(
    config.initial_collect_steps))
  gin_bindings.append('ENV_STR = "{}"'.format(config.env_str))
  gin_bindings.append('NUM_STEPS = {}'.format(config.num_steps))
  gin_bindings.append('LAYER_SIZE = {}'.format(config.layer_size))
  return gin_bindings


def wandb_log_callback(summaries, step=None):
  del step
  for summary in summaries:
    wandb.tensorflow.log(summary)


def main(_):
  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))
  config = wandb.config
  if not wandb.run.resumed:  # do not make changes
    root_path = []
    if os.environ.get('EXP_DIR'):
      root_path.append(os.environ.get('EXP_DIR'))
    root_path.append(config.root_dir)
    root_path.append(str(os.environ.get('WANDB_RUN_ID', 0)))
    config.update(dict(root_dir=osp.join(*root_path)), allow_val_change=True)
  gin_files = config.gin_files
  gin_bindings = gin_bindings_from_config(config)
  gin.parse_config_files_and_bindings(gin_files, gin_bindings)
  # tf.config.threading.set_inter_op_parallelism_threads(12)
  trainer.train_eval(config.root_dir, batch_size=config.batch_size, seed=FLAGS.seed,
                     train_metrics_callback=wandb.log, eager_debug=FLAGS.eager_debug)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
