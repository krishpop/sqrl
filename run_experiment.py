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
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'Name for run on wandb')
flags.DEFINE_string('root_dir', 'sac-sweeps/cube_rotate/', 'Root directory for writing logs/summaries/checkpoints')
flags.DEFINE_string('load_dir', None, 'Directory for loading pretrained policy')
flags.DEFINE_string('load_run', None, 'Loads wandb and gin configs from past run')
flags.DEFINE_string('env_str', 'SafemrlCube-v2', 'Environment string')
flags.DEFINE_boolean('monitor', False, 'load environments with Monitor wrapper')
flags.DEFINE_boolean('finetune', False, 'Fine-tuning task safety')
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps')
flags.DEFINE_integer('layer_size', 256, 'Number of training steps')
flags.DEFINE_integer('batch_size', 256, 'batch size used for training')
flags.DEFINE_float('safety_gamma', 0.7, 'Safety discount term used for TD backups')
flags.DEFINE_float('target_safety', 0.1, 'Target safety for safety critic')
flags.DEFINE_integer('target_entropy', None, 'Target entropy for policy')
flags.DEFINE_integer('n_critics', None, 'number of critics to use')
flags.DEFINE_float('friction', None, 'update minitaur friction param')
flags.DEFINE_float('drop_penalty', -500., 'Drop penalty for cube environment')
flags.DEFINE_float('lr', None, 'Learning rate for all optimizers')
flags.DEFINE_float('actor_lr', 3e-4, 'Learning rate for actor')
flags.DEFINE_float('critic_lr', 3e-4, 'Learning rate for critic')
flags.DEFINE_float('entropy_lr', 3e-4, 'Learning rate for alpha')
flags.DEFINE_float('safety_lr', 3e-4, 'Learning rate for safety critic')
flags.DEFINE_float('target_update_tau', 0.001, 'Factor for soft update of the target networks')
flags.DEFINE_integer('target_update_period', 1, 'Period for soft update of the target networks')
flags.DEFINE_integer('initial_collect_steps', 2000, 'Number of steps to collect with random policy')
flags.DEFINE_float('initial_log_alpha', 0., 'Initial value for log_alpha')
flags.DEFINE_float('gamma', 0.99, 'Future reward discount factor')
flags.DEFINE_float('reward_scale_factor', 1.0, 'Reward scale factor for SacAgent')
flags.DEFINE_float('gradient_clipping', 2., 'Gradient clipping factor for SacAgent')
flags.DEFINE_multi_string('gin_files', ['sac.gin', 'cube_default.gin'],
                          'gin files to load')
flags.DEFINE_boolean('debug_summaries', False, 'Debug summaries for critic and actor')
flags.DEFINE_boolean('debug', False, 'Debug logging')
flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
flags.DEFINE_integer('seed', None, 'Seed to seed envs and algorithm with')

#wandb.init(sync_tensorboard=True, entity='krshna', project='safemrl-2', config=FLAGS,
#           monitor_gym=True)

# blacklist of config values to update with a loaded run config
RUN_CONFIG_BLACKLIST = {'safety_gamma', 'target_safety', 'friction', 'drop_penalty',
                        'target_entropy', 'env_str', 'root_dir', 'num_steps', 'finetune'}


def gin_bindings_from_config(config):
  gin_bindings = []
  if 'sac_safe_online.gin' in config.gin_files:
    gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.safety_gamma = {}'.format(config.safety_gamma))
    gin_bindings.append('safe_sac_agent.SafeSacAgentOnline.target_safety = {}'.format(config.target_safety))
    agent_prefix = 'safe_sac_agent.SafeSacAgentOnline'
  elif 'sac.gin' in config.gin_files:
    agent_prefix = 'sac_agent.SacAgent'
  elif 'wcpg.gin' in config.gin_files:
    agent_prefix = 'wcpg_agent.WcpgAgent'
  elif 'sac_ensemble.gin' in config.gin_files:
    gin_bindings.append('trainer.train_eval.n_critics = {}'.format(config.n_critics))
    agent_prefix = 'ensemble_sac_agent.EnsembleSacAgent'

  gin_bindings.append('{}.reward_scale_factor = {}'.format(agent_prefix, config.reward_scale_factor))
  gin_bindings.append('{}.target_update_tau = {}'.format(agent_prefix, config.target_update_tau))
  gin_bindings.append('{}.target_update_period = {}'.format(agent_prefix, config.target_update_period))
  gin_bindings.append('{}.gamma = {}'.format(agent_prefix, config.gamma))
  gin_bindings.append('{}.gradient_clipping = {}'.format(agent_prefix, config.gradient_clipping))

  ###### AGENT-SPECIFIC bindings
  if agent_prefix != 'wcpg_agent.WcpgAgent':  # WCPG does not use target entropy
    gin_bindings.append('{}.target_entropy = {}'.format(agent_prefix, config.target_entropy))
    if config.entropy_lr:
      gin_bindings.append('al_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.entropy_lr))

  if agent_prefix.split('.')[0] == 'safe_sac_agent' and config.safety_lr:
    gin_bindings.append('sc_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.safety_lr))

  if agent_prefix == 'ensemble_sac_agent.EnsembleSacAgent':
    gin_bindings.append('trainer.train_eval.critic_learning_rate = {}'.format(config.critic_lr))
  elif config.critic_lr:
    gin_bindings.append('cr_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.critic_lr))

  if config.lr and not wandb.run.resumed:  # do not update config if resuming run
    if config.lr < 0:  # HACK: hp.loguniform not working
      config.update(dict(lr=10 ** config.lr), allow_val_change=True)
    gin_bindings.append('LEARNING_RATE = {}'.format(config.lr))
  elif not wandb.run.resumed:
    if config.actor_lr and config.actor_lr < 0:
      config.update(dict(actor_lr=10**config.actor_lr), allow_val_change=True)
    if config.critic_lr and config.critic_lr < 0:
      config.update(dict(critic_lr=10**config.critic_lr), allow_val_change=True)
    if config.entropy_lr and config.entropy_lr < 0:
      config.update(dict(entropy_lr=10**config.entropy_lr), allow_val_change=True)

  ######## ENV-SPECIFIC BINDINGS
  if 'Minitaur' in config.env_str and config.friction:
    gin_bindings.append('minitaur.MinitaurGoalVelocityEnv.friction = {}'.format(config.friction))
  elif 'Cube' in config.env_str and config.drop_penalty:
    gin_bindings.append('cube_env.SafemrlCubeEnv.drop_penalty = {}'.format(config.drop_penalty))

  gin_bindings.append('ac_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.actor_lr))

  gin_bindings.append("INITIAL_NUM_STEPS = {}".format(config.initial_collect_steps))
  gin_bindings.append('ENV_STR = "{}"'.format(config.env_str))
  # HACK, to prevent num_steps being overwritten during loading/resuming a previous run.
  gin_bindings.append('NUM_STEPS = {}'.format(config.num_steps))
  gin_bindings.append('LAYER_SIZE = {}'.format(config.layer_size))
  return gin_bindings


def wandb_log_callback(summaries, step=None):
  del step
  for summary in summaries:
    wandb.tensorflow.log(summary)


def main(_):
  logging.set_verbosity(logging.INFO)
  wandb.init(name=FLAGS.name, sync_tensorboard=True, entity='krshna', project='safemrl-2', config=FLAGS,
             monitor_gym=True)
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)
  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))
  config = wandb.config
  if not wandb.run.resumed:  # do not make changes
    root_path = []
    exp_dir = os.environ.get('EXP_DIR')
    if exp_dir and exp_dir not in FLAGS.root_dir and not osp.exists(FLAGS.root_dir):
      root_path.append(exp_dir)
    root_path.append(FLAGS.root_dir)
    root_path.append(str(os.environ.get('WANDB_RUN_ID', 0)))
    config.update(dict(root_dir=osp.join(*root_path)), allow_val_change=True)
    config.update(dict(num_steps=FLAGS.num_steps), allow_val_change=True)
  gin_bindings = []
  if FLAGS.load_run:
    api = wandb.Api(overrides=dict(entity='krshna', project='safemrl-2'))
    run = api.run(path=FLAGS.load_run)
    op_config = os.path.join(run.config['root_dir'], 'train/operative_config-0.gin')
    config.update(dict(gin_files=[op_config]), allow_val_change=True)
    config.update({k: run.config[k] for k in run.config if k not in RUN_CONFIG_BLACKLIST}, allow_val_change=True)
    if 'Cube' in config.env_str and config.finetune:
      # If fine-tuning, make task more difficult
      gin_bindings.append("cube_env.SafemrlCubeEnv.goal_task = ('more_left', 'more_right', 'more_up', 'more_down')")
  gin_bindings += gin_bindings_from_config(config)
  gin.parse_config_files_and_bindings(config.gin_files, gin_bindings)
  # tf.config.threading.set_inter_op_parallelism_threads(12)
  trainer.train_eval(config.root_dir, load_root_dir=FLAGS.load_dir, batch_size=config.batch_size, seed=FLAGS.seed,
                     train_metrics_callback=wandb.log, eager_debug=FLAGS.eager_debug,
                     monitor=FLAGS.monitor, debug_summaries=FLAGS.debug_summaries, pretraining=(not FLAGS.finetune))


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
