import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
import wandb
import gin
import trainer
import os
import os.path as osp

from absl import flags
from absl import app
from absl import logging

# blacklist of config values to update with a loaded run config
RUN_CONFIG_BLACKLIST = {'safety_gamma', 'target_safety', 'friction',
                        'goal_vel', 'action_noise', 'action_scale',
                        'target_entropy', 'root_dir', 'num_steps', 'finetune',
                        'debug_summaries', 'train_finetune', 'finetune_steps',
                        'eager_debug', 'debug', 'num_threads'}

# keys to exclude from wandb config
FLAGS = flags.FLAGS
EXCLUDE_KEYS = list(FLAGS) + ['name', 'notes', 'monitor', 'debug', 'eager_debug',
                              'num_threads', 'help', 'helpfull', 'helpshort', 'helpxml',
                              'resume_id']

def define_flags():
  # Configs & checkpointing args
  flags.DEFINE_string('name', None, 'Name for run on wandb')
  flags.DEFINE_string('notes', None, 'Notes for describing run')
  flags.DEFINE_string('root_dir', None, 'Root directory for writing logs/summaries/checkpoints')
  flags.DEFINE_string('load_dir', None, 'Directory for loading pretrained policy')
  flags.DEFINE_string('load_run', None, 'Loads wandb and gin configs from past run')
  flags.DEFINE_boolean('load_config', False, 'whether or not to load config with run')
  flags.DEFINE_multi_string('gin_files', None,
                            'gin files to load')
  flags.DEFINE_string('resume_id', None, 'enables loading config and everything from previous run')
  flags.DEFINE_multi_string('gin_param', None, 'params to add to gin_bindings')

  # Trainer args
  flags.DEFINE_string('env_str', None, 'Environment string')
  flags.DEFINE_boolean('monitor', False, 'load environments with Monitor wrapper')
  flags.DEFINE_boolean('finetune', False, 'Fine-tuning task safety')
  flags.DEFINE_boolean('finetune_sc', True, 'Fine-tuning safety critic')
  flags.DEFINE_boolean('train_finetune', False, 'Train and immediately finetune')
  flags.DEFINE_float('finetune_steps', 500000, 'Number of finetuning steps')
  flags.DEFINE_float('num_steps', None, 'Number of training steps')
  flags.DEFINE_integer('initial_collect_steps', None, 'Number of steps to collect with random policy')
  flags.DEFINE_boolean('offline', False, 'Whether to train safety critic online')
  flags.DEFINE_boolean('debug_summaries', False, 'Debug summaries for critic and actor')
  flags.DEFINE_boolean('debug', False, 'Debug logging')
  flags.DEFINE_boolean('eager_debug', False, 'Debug in eager mode if True')
  flags.DEFINE_integer('seed', None, 'Seed to seed envs and algorithm with')
  flags.DEFINE_integer('num_threads', None, 'Max number of threads for TF to spin up')
  flags.DEFINE_integer('batch_size', 256, 'batch size used for training')

  # Model args
  flags.DEFINE_integer('layer_size', None, 'Number of training steps')
  flags.DEFINE_float('init_means_output_factor', None, 'Action initialization scale')
  flags.DEFINE_float('std_bias_init_value', None, 'Actor stddev bias initial value')

  # Algorithm args
  flags.DEFINE_float('lr', None, 'Learning rate for all optimizers')
  flags.DEFINE_float('actor_lr', None, 'Learning rate for actor')
  flags.DEFINE_float('critic_lr', None, 'Learning rate for critic')
  flags.DEFINE_float('target_update_tau', None, 'Factor for soft update of the target networks')
  flags.DEFINE_integer('target_update_period', None, 'Period for soft update of the target networks')
  flags.DEFINE_float('gamma', None, 'Future reward discount factor')
  flags.DEFINE_float('reward_scale_factor', None, 'Reward scale factor for SacAgent')
  flags.DEFINE_float('gradient_clipping', None, 'Gradient clipping factor for SacAgent')
  flags.DEFINE_integer('lambda_schedule_nsteps', None, 'Use linear lambda scheduler')
  flags.DEFINE_float('lambda_initial', None, 'sets initial lambda value')
  flags.DEFINE_float('lambda_final', None, 'Final lambda value (if using scheduler)')

  ## SAC args
  flags.DEFINE_float('entropy_lr', None, 'Learning rate for alpha')
  flags.DEFINE_integer('target_entropy', None, 'Target entropy for policy')
  ### SQRL args
  flags.DEFINE_float('safety_lr', None, 'Learning rate for safety critic')
  flags.DEFINE_float('safety_gamma', None, 'Safety discount term used for TD backups')
  flags.DEFINE_float('target_safety', None, 'Target safety for safety critic')
  ### SAC-ensemble args
  flags.DEFINE_integer('num_critics', None, 'number of critics to use')
  flags.DEFINE_float('percentile', None, 'ensemble percentile')

  # Env args
  ## Minitaur
  flags.DEFINE_float('friction', None, 'Friction for Minitaur environment')
  flags.DEFINE_float('goal_vel', None, 'Goal velocity for Minitaur environment')
  ## PointMass
  flags.DEFINE_string('pm_goal', '6,3', 'PointMass goal location, string of comma-separated integers within boundaries of goal range')
  flags.DEFINE_float('action_noise', None, 'Action noise for point-mass environment')
  flags.DEFINE_float('action_scale', None, 'Action scale for point-mass environment')


define_flags()


def load_prev_run(config):
  api = wandb.Api(overrides=dict(entity='krshna', project='sqrl-neurips'))
  run = api.run(path=FLAGS.load_run)
  # Find correct load_path, invariant to which machine training was done on
  exp_dir = os.environ.get('EXP_DIR')
  if 'tfagents' in run.config['root_dir']:
    root_dir = run.config['root_dir'].split('tfagents/')[-1]
  elif 'data' in run.config['root_dir']:
    root_dir = run.config['root_dir'].split('data/')[-1]
  load_path = osp.join(exp_dir, root_dir)
  assert osp.exists(load_path), 'tried to load path the does not exist: {}'.format(load_path)
  op_config = os.path.join(load_path, 'train/operative_config-0.gin')
  # if resuming run without finetuning, adds operative config to config list
  if wandb.run.resumed and not FLAGS.finetune:
    config.update(dict(gin_files=run.config['gin_files'] + [op_config]), allow_val_change=True)
  # copies rest of loaded run config excluding blacklist
  if config.load_config or wandb.run.resumed:
    config.update({k: run.config[k] for k in run.config if k not in RUN_CONFIG_BLACKLIST},
                  allow_val_change=True)


def update_root(config):
  root_path = []
  exp_dir = os.environ.get('EXP_DIR')
  if exp_dir and exp_dir not in FLAGS.root_dir and not osp.exists(FLAGS.root_dir):
    root_path.append(exp_dir)
  root_path.append(FLAGS.root_dir)
  root_path.append(str(os.environ.get('WANDB_RUN_ID', 0)))
  config.update(dict(root_dir=osp.join(*root_path)), allow_val_change=True)


def gin_bindings_from_config(config, gin_bindings=[]):
  # TODO: turn all configurable gin bindings into macros or REMOVE

  gin_bindings = gin_bindings or []
  agent_class = gin.query_parameter('%AGENT_CLASS')
  logging.info("Agent class: {}".format(agent_class))
  # Configure agent prefixes
  if agent_class == 'sqrl':
    agent_prefix = 'safe_sac_agent.SqrlAgent'
    if config.safety_gamma:
      gin_bindings.append('SAFETY_GAMMA = {}'.format(config.safety_gamma))
    if config.target_safety:
      gin_bindings.append('TARGET_SAFETY = {}'.format(config.target_safety))
    if config.offline:
      gin_bindings.append('trainer.train_eval.online_critic = False')
  elif agent_class == 'sac':
    agent_prefix = 'sac_agent.SacAgent'
  elif agent_class == 'wcpg':
    agent_prefix = 'wcpg_agent.WcpgAgent'
  elif agent_class == 'sac_ensemble':
    if not wandb.run.resumed:
      if config.num_critics:
        gin_bindings.append('trainer.train_eval.num_critics = {}'.format(config.num_critics))
      if config.percentile:
        gin_bindings.append('ensemble_sac_agent.EnsembleSacAgent.percentile = {}'.format(config.percentile))
    agent_prefix = 'ensemble_sac_agent.EnsembleSacAgent'

  # Config value updates
  if not wandb.run.resumed:
    if config.lr:
      # do not update config learning rate if resuming run
      if config.lr < 0:  # HACK: hp.loguniform not working
        config.update(dict(lr=10 ** config.lr), allow_val_change=True)
      gin_bindings.append('LEARNING_RATE = {}'.format(config.lr))
    else:
      if config.actor_lr and config.actor_lr < 0:
        ac_lr = round(10 ** config.actor_lr, 5)
        config.update(dict(actor_lr=ac_lr), allow_val_change=True)
      if config.critic_lr and config.critic_lr < 0:
        cr_lr = round(10 ** config.critic_lr, 5)
        config.update(dict(critic_lr=cr_lr), allow_val_change=True)
      if config.entropy_lr and config.entropy_lr < 0:
        al_lr = round(10 ** config.entropy_lr, 5)
        config.update(dict(entropy_lr=al_lr), allow_val_change=True)
      if config.safety_lr and config.safety_lr < 0:
        sc_lr = round(10 ** config.safety_lr, 5)
        config.update(dict(entropy_lr=sc_lr), allow_val_change=True)

    # Generic agent bindings
    if config.init_means_output_factor:
      gin_bindings.append('agents.normal_projection_net.init_means_output_factor = {}'.format(
        config.init_means_output_factor))
    if config.std_bias_init_value:
      gin_bindings.append('agents.normal_projection_net.std_bias_initializer_value = {}'.format(
        config.std_bias_init_value
      ))

    if config.reward_scale_factor:
      gin_bindings.append('{}.reward_scale_factor = {}'.format(agent_prefix, config.reward_scale_factor))
    if config.target_update_tau:
      gin_bindings.append('{}.target_update_tau = {}'.format(agent_prefix, config.target_update_tau))
    if config.target_update_period:
      gin_bindings.append('{}.target_update_period = {}'.format(agent_prefix, config.target_update_period))
    if config.gamma:
      gin_bindings.append('{}.gamma = {}'.format(agent_prefix, config.gamma))
    gin_bindings.append('{}.gradient_clipping = {}'.format(agent_prefix, config.gradient_clipping))

  ## Agent-specific bindings
  if agent_prefix != 'wcpg_agent.WcpgAgent':  # WCPG does not use target entropy
    if config.target_entropy:
      gin_bindings.append('{}.target_entropy = {}'.format(agent_prefix, config.target_entropy))
    if config.entropy_lr and not wandb.run.resumed:
      gin_bindings.append('al_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.entropy_lr))

  if not wandb.run.resumed:
    if config.safety_lr and agent_class in ['sqrl']:
      gin_bindings.append('sc_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.safety_lr))
    if config.critic_lr:
      if agent_prefix == 'ensemble_sac_agent.EnsembleSacAgent':
        gin_bindings.append('trainer.train_eval.critic_learning_rate = {}'.format(config.critic_lr))
      else:
        gin_bindings.append('cr_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.critic_lr))
    if config.actor_lr:
      gin_bindings.append('ac_opt/tf.keras.optimizers.Adam.learning_rate = {}'.format(config.actor_lr))

  ## Env-specific bindings
  env_str = config.env_str or gin.query_parameter('%ENV_STR')
  if 'Minitaur' in env_str:
    if config.friction:
      gin_bindings.append('minitaur.MinitaurGoalVelocityEnv.friction = {}'.format(config.friction))
    if config.goal_vel:
      gin_bindings.append("minitaur.MinitaurGoalVelocityEnv.goal_vel = {}".format(config.goal_vel))
  elif 'Cube' in env_str:
    if config.finetune:
      gin_bindings.append("cube_env.SafemrlCubeEnv.goal_task = ('more_left', 'more_right', 'more_up', 'more_down')")
  elif 'DrunkSpider' in env_str:
    if config.action_noise:
      gin_bindings.append('point_mass.PointMassEnv.action_noise = {}'.format(config.action_noise))
    if config.action_scale:
      gin_bindings.append('point_mass.PointMassEnv.action_scale = {}'.format(config.action_scale))
    if config.finetune:
      gin_bindings.append("point_mass.env_load_fn.goal = ({}})".format(config.pm_goal))

  if config.initial_collect_steps:
    gin_bindings.append("INITIAL_NUM_STEPS = {}".format(config.initial_collect_steps))
  if config.env_str:
    gin_bindings.append('ENV_STR = "{}"'.format(config.env_str))
  if FLAGS.num_steps:
    gin_bindings.append('NUM_STEPS = {}'.format(int(FLAGS.num_steps)))
  if config.layer_size and not wandb.run.resumed:
    gin_bindings.append('LAYER_SIZE = {}'.format(config.layer_size))
  return gin_bindings


def finetune_gin_bindings(config):
  gin_bindings = []
  env_str = config.env_str or gin.query_parameter('%ENV_STR')
  if 'Minitaur' in env_str:
    if config.friction:
      gin_bindings.append('minitaur.MinitaurGoalVelocityEnv.friction = {}'.format(config.friction))
    if config.goal_vel:
      gin_bindings.append("minitaur.MinitaurGoalVelocityEnv.goal_vel = {}".format(config.goal_vel))
  elif 'Cube' in env_str:
    if config.finetune:
      gin_bindings.append("cube_env.SafemrlCubeEnv.goal_task = ('more_left', 'more_right', 'more_up', 'more_down')")
  elif 'DrunkSpider' in env_str:
    if config.action_noise:
      gin_bindings.append('point_mass.PointMassEnv.action_noise = {}'.format(config.action_noise))
    if config.action_scale:
      gin_bindings.append('point_mass.PointMassEnv.action_scale = {}'.format(config.action_scale))
    if config.finetune:
      gin_bindings.append("point_mass.env_load_fn.goal = (6, 3)")
  # set NUM_STEPS to be previous value + FINETUNE_STEPS
  ft_steps = gin.query_parameter("%FINETUNE_STEPS")
  if ft_steps is None:
    ft_steps = config.finetune_steps
  num_steps = gin.query_parameter("%NUM_STEPS") + ft_steps
  gin_bindings.append('NUM_STEPS = {}'.format(num_steps))
  return gin_bindings


def wandb_log_callback(summaries, step=None):
  del step
  for summary in summaries:
    wandb.tensorflow.log(summary)


def main(_):
  name = FLAGS.name
  if FLAGS.seed is not None and name:
    name = '-'.join([name, str(FLAGS.seed)])
  run = wandb.init(name=name, sync_tensorboard=True, entity='krshna', project='sqrl-neurips',
                   config=FLAGS, monitor_gym=FLAGS.monitor, config_exclude_keys=EXCLUDE_KEYS,
                   notes=FLAGS.notes, resume=FLAGS.resume_id)

  logging.set_verbosity(logging.INFO)
  if FLAGS.debug:
    logging.set_verbosity(logging.DEBUG)

  if os.environ.get('CONFIG_DIR'):
    gin.add_config_file_search_path(os.environ.get('CONFIG_DIR'))

  config = wandb.config

  # Only update root_path if not resuming a run
  if not wandb.run.resumed:
    update_root(config)

  if FLAGS.load_run:
    load_prev_run(config)

  gin_bindings = FLAGS.gin_param or []

  for gin_file in config.gin_files:
    if gin_file == 'sac_safe_online.gin':
      gin_file = 'sqrl.gin'
    gin.parse_config_file(gin_file, [])

  gin_bindings = gin_bindings_from_config(config) + gin_bindings
  gin.parse_config_files_and_bindings([], gin_bindings)

  if FLAGS.num_threads:
    tf.config.threading.set_inter_op_parallelism_threads(FLAGS.num_threads)

  trainer.train_eval(config.root_dir, load_root_dir=FLAGS.load_dir,
                     batch_size=config.batch_size,
                     seed=FLAGS.seed, train_metrics_callback=wandb.log,
                     eager_debug=FLAGS.eager_debug,
                     monitor=FLAGS.monitor,
                     debug_summaries=FLAGS.debug_summaries,
                     pretraining=(not FLAGS.finetune),
                     finetune_sc=FLAGS.finetune_sc, wandb=True)

  if config.train_finetune and not config.finetune:
    with gin.unlock_config():
      finetune_bindings = finetune_gin_bindings(config)
      gin.parse_config_files_and_bindings([], finetune_bindings)
    trainer.train_eval(config.root_dir, load_root_dir=FLAGS.load_dir,
                       batch_size=config.batch_size,
                       seed=FLAGS.seed, train_metrics_callback=wandb.log,
                       eager_debug=FLAGS.eager_debug,
                       monitor=FLAGS.monitor,
                       debug_summaries=FLAGS.debug_summaries,
                       pretraining=False,
                       finetune_sc=FLAGS.finetune_sc, wandb=True)

if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
