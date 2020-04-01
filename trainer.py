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

r"""Train and Eval Safety-Constrained SAC."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import os.path as osp
import time

from absl import logging

import gin
import gym
import tensorflow as tf
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common, nest_utils

from safemrl.algos import agents
from safemrl.algos import safe_sac_agent
from safemrl.algos import ensemble_sac_agent
from safemrl.algos import wcpg_agent
from safemrl.utils import safe_dynamic_episode_driver
from safemrl.utils import misc
from safemrl.utils import metrics

try:
  import highway_env
except ImportError:
  logging.debug("Could not import highway_env")

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9

SAFETY_ENVS = ['IndianWell', 'IndianWell2', 'IndianWell3', 'DrunkSpider', 'pddm_cube',
               'SafemrlCube', 'highway',
               'DrunkSpiderShort', 'MinitaurGoalVelocityEnv', 'MinitaurRandFrictionGoalVelocityEnv']
SAFETY_AGENTS = [safe_sac_agent.SafeSacAgentOnline]
ALGOS = {'sac': sac_agent.SacAgent,
         'sac_safe_online': safe_sac_agent.SafeSacAgentOnline,
         'sac_ensemble': ensemble_sac_agent.EnsembleSacAgent,
         'wcpg': wcpg_agent.WcpgAgent}

# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_LOSS_STEPS = 100


@gin.configurable(blacklist=['seed', 'eager_debug', 'monitor', 'debug_summaries'])
def train_eval(
    root_dir,
    load_root_dir=None,
    env_load_fn=None,
    gym_env_wrappers=[],
    monitor=False,
    env_name=None,
    agent_class=None,
    initial_collect_driver_class=None,
    collect_driver_class=None,
    online_driver_class=dynamic_episode_driver.DynamicEpisodeDriver,
    num_global_steps=1000000,
    train_steps_per_iteration=1,
    train_metrics=None,
    eval_metrics=None,
    train_metrics_callback=None,
    # Params for SacAgent args
    actor_fc_layers=(256, 256),
    critic_joint_fc_layers=(256, 256),
    # Safety Critic training args
    train_sc_steps=10,
    train_sc_interval=1000,
    online_critic=False,
    n_envs=None,
    finetune_sc=False,
    pretraining=True,
    # Ensemble Critic training args
    n_critics=30,
    critic_learning_rate=3e-4,
    # Wcpg Critic args
    critic_preprocessing_layer_size=256,
    actor_preprocessing_layer_size=256,
    # Params for train
    batch_size=256,
    # Params for eval
    run_eval=False,
    num_eval_episodes=10,
    max_episode_len=500,
    eval_interval=1000,
    eval_metrics_callback=None,
    # Params for summaries and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=50000,
    keep_rb_checkpoint=False,
    log_interval=1000,
    summary_interval=1000,
    monitor_interval=5000,
    summaries_flush_secs=10,
    early_termination_fn=None,
    debug_summaries=False,
    seed=None,
    eager_debug=False,
    env_metric_factories=None,
    wandb=False):  # pylint: disable=unused-argument
  """A simple train and eval for SC-SAC."""
  if isinstance(agent_class, str):
    assert agent_class in ALGOS, 'trainer.train_eval: agent_class {} invalid'.format(agent_class)
    agent_class = ALGOS.get(agent_class)
  n_envs = n_envs or num_eval_episodes
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  train_metrics = train_metrics or []
  eval_metrics = eval_metrics or []
  sc_metrics = eval_metrics or []

  updating_sc = online_critic and (not load_root_dir or finetune_sc)
  logging.debug('updating safety critic: {}'.format(updating_sc))

  if seed:
    tf.compat.v1.set_random_seed(seed)

  if agent_class in SAFETY_AGENTS:
    sc_dir = os.path.join(root_dir, 'sc')
    if online_critic:
      sc_summary_writer = tf.compat.v2.summary.create_file_writer(
        sc_dir, flush_millis=summaries_flush_secs * 1000)
      sc_metrics = [
          tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes, batch_size=n_envs, name='SafeAverageReturn'),
          tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes, batch_size=n_envs,
              name='SafeAverageEpisodeLength')] + [tf_py_metric.TFPyMetric(m) for m in sc_metrics]
      sc_tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
          [lambda: env_load_fn(env_name, gym_env_wrappers=gym_env_wrappers)] * n_envs
        ))
    if seed:
      try:
        for i, pyenv in enumerate(sc_tf_env.pyenv.envs):
          pyenv.seed(seed * n_envs + i)
      except:
        pass

  if run_eval:
    eval_dir = os.path.join(root_dir, 'eval')
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes, batch_size=n_envs),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes, batch_size=n_envs),
    ] + [tf_py_metric.TFPyMetric(m) for m in eval_metrics]
    eval_tf_env = tf_py_environment.TFPyEnvironment(
      parallel_py_environment.ParallelPyEnvironment(
        [lambda: env_load_fn(env_name, gym_env_wrappers=gym_env_wrappers)] * n_envs
      ))
    if seed:
      try:
        for i, pyenv in enumerate(eval_tf_env.pyenv.envs):
          pyenv.seed(seed * n_envs + i)
      except:
        pass
  elif 'Drunk' in env_name:
    # Just visualizes trajectories
    eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))

  if monitor:
    vid_path = os.path.join(root_dir, 'rollouts')
    monitor_env_wrapper = misc.monitor_freq(1, vid_path)
    monitor_env = gym.make(env_name)
    for wrapper in gym_env_wrappers:
      monitor_env = wrapper(monitor_env)
    monitor_env = monitor_env_wrapper(monitor_env)
    # auto_reset must be False to ensure Monitor works correctly
    monitor_py_env = gym_wrapper.GymWrapper(monitor_env, auto_reset=False)

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    py_env = env_load_fn(env_name, gym_env_wrappers=gym_env_wrappers)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    if seed:
      try:
        for i, pyenv in enumerate(tf_env.pyenv.envs):
          pyenv.seed(seed * n_envs + i)
      except:
        pass
    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()

    logging.debug('obs spec: %s', observation_spec)
    logging.debug('action spec: %s', action_spec)

    # tf.summary.trace_on()

    if agent_class == wcpg_agent.WcpgAgent:
      alpha_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, minimum=0., maximum=1.,
                                                 name='alpha')
      input_tensor_spec = (observation_spec['observation'], action_spec, alpha_spec)
      logging.debug('input_tensor_spec: %s', input_tensor_spec)
      critic_net = agents.DistributionalCriticNetwork(
        input_tensor_spec, preprocessing_layer_size=critic_preprocessing_layer_size,
        joint_fc_layer_params=critic_joint_fc_layers)
      actor_net = agents.WcpgActorNetwork((observation_spec['observation'], alpha_spec), action_spec)
    else:
      actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=agents.normal_projection_net)
      critic_net = agents.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=critic_joint_fc_layers)

    if agent_class in SAFETY_AGENTS:
      logging.debug('making safety agent')
      safety_critic_net = agents.CriticNetwork(
          (observation_spec, action_spec),
          joint_fc_layer_params=critic_joint_fc_layers)
      tf_agent = agent_class(
          time_step_spec,
          action_spec,
          actor_network=actor_net,
          critic_network=critic_net,
          safety_critic_network=safety_critic_net,
          train_step_counter=global_step,
          debug_summaries=debug_summaries,
          safety_pretraining=pretraining,
          train_critic_online=online_critic)
    elif agent_class is ensemble_sac_agent.EnsembleSacAgent:
      critic_nets, critic_optimizers = [critic_net], [tf.keras.optimizers.Adam(critic_learning_rate)]
      for _ in range(n_critics-1):
        critic_nets.append(agents.CriticNetwork((observation_spec, action_spec),
                                                joint_fc_layer_params=critic_joint_fc_layers))
        critic_optimizers.append(tf.keras.optimizers.Adam(critic_learning_rate))
      tf_agent = agent_class(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_networks=critic_nets,
        critic_optimizers=critic_optimizers,
        debug_summaries=debug_summaries
      )
    else:  # assume is using SacAgent or WCPG
      logging.debug(critic_net.input_tensor_spec)
      tf_agent = agent_class(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        train_step_counter=global_step,
        debug_summaries=debug_summaries)

    tf_agent.initialize()

    # Make the replay buffer.
    collect_data_spec = tf_agent.collect_data_spec

    logging.debug('Allocating replay buffer ...')
    # Add to replay buffer and other agent specific observers.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      collect_data_spec,
      batch_size=1,
      max_length=1000000)

    logging.debug('RB capacity: %i', replay_buffer.capacity)
    logging.debug('ReplayBuffer Collect data spec: %s', collect_data_spec)

    agent_observers = [replay_buffer.add_batch]
    if agent_class in SAFETY_AGENTS:  # online_critic:
      online_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
          collect_data_spec, batch_size=1, max_length=500000, dataset_window_shift=1)
      # agent_observers.append(online_replay_buffer.add_batch)  # collect policy only adds to online_rb

      online_rb_ckpt_dir = os.path.join(train_dir, 'online_replay_buffer')
      online_rb_checkpointer = common.Checkpointer(
          ckpt_dir=online_rb_ckpt_dir,
          max_to_keep=1,
          replay_buffer=online_replay_buffer)

      # clear_rb = online_replay_buffer.clear

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
    ] + [tf_py_metric.TFPyMetric(m) for m in train_metrics]

    # SETUP TF AGENTS POLICIES
    if not online_critic:
      eval_policy = tf_agent.policy
      collect_policy = tf_agent.collect_policy
    else:
      eval_policy = tf_agent._safe_policy
      collect_policy = tf_agent.collect_policy
      online_collect_policy = tf_agent._safe_policy #  if pretraining else tf_agent.collect_policy

    if not load_root_dir:
      initial_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)
    else:
      initial_collect_policy = collect_policy
    if agent_class == wcpg_agent.WcpgAgent:
      initial_collect_policy = agents.WcpgPolicyWrapper(initial_collect_policy)

    # tf.summary.trace_off()

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)
    if agent_class in SAFETY_AGENTS:
      safety_critic_checkpointer = common.Checkpointer(
          ckpt_dir=sc_dir,
          safety_critic=tf_agent._safety_critic_network,  # pylint: disable=protected-access
          global_step=global_step)
    rb_ckpt_dir = os.path.join(train_dir, 'replay_buffer')
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=rb_ckpt_dir, max_to_keep=1, replay_buffer=replay_buffer)

    if load_root_dir:
      load_root_dir = os.path.expanduser(load_root_dir)
      load_train_dir = os.path.join(load_root_dir, 'train')
      misc.load_agent_ckpt(load_train_dir, tf_agent)
      # if len(os.listdir(os.path.join(load_train_dir, 'replay_buffer'))) > 1:
      #   load_rb_ckpt_dir = os.path.join(load_train_dir, 'replay_buffer')
      #   misc.load_rb_ckpt(load_rb_ckpt_dir, replay_buffer)
      online_load_rb_ckpt_dir = os.path.join(load_train_dir, 'online_replay_buffer')
      if osp.exists(online_load_rb_ckpt_dir):
        misc.load_rb_ckpt(online_load_rb_ckpt_dir, online_replay_buffer)

      # TODO: REMOVE THIS, HARDCODED
      lr = 5e-3
      sac_lr = 3e-4
      if agent_class in SAFETY_AGENTS:
        tf_agent._lambda_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
      if agent_class != wcpg_agent.WcpgAgent:
        tf_agent._alpha_optimizer.learning_rate = lr
      if agent_class in SAFETY_AGENTS:
        tf_agent._safety_critic_optimizer.learning_rate = lr
      if agent_class != ensemble_sac_agent.EnsembleSacAgent:
        tf_agent._critic_optimizer = tf.keras.optimizers.Adam(learning_rate=sac_lr)
        tf_agent._actor_optimizer = tf.keras.optimizers.Adam(learning_rate=sac_lr)

    if load_root_dir is None:
      train_checkpointer.initialize_or_restore()
      rb_checkpointer.initialize_or_restore()

    if agent_class in SAFETY_AGENTS:
      safety_critic_checkpointer.initialize_or_restore()

    env_metrics = []
    if env_metric_factories:
      for env_metric in env_metric_factories:
        env_metrics.append(tf_py_metric.TFPyMetric(env_metric([py_env.gym])))
        # if run_eval:
        #   eval_metrics.append(env_metric([env.gym for env in eval_tf_env.pyenv._envs]))
        # if online_critic:
        #   sc_metrics.append(env_metric([env.gym for env in sc_tf_env.pyenv._envs]))

    collect_driver = collect_driver_class(
        tf_env, collect_policy, observers=agent_observers + train_metrics + env_metrics)

    if online_critic:
      logging.debug('online driver class: %s', online_driver_class)
      if online_driver_class is safe_dynamic_episode_driver.SafeDynamicEpisodeDriver:
        online_temp_buffer = episodic_replay_buffer.EpisodicReplayBuffer(collect_data_spec)
        online_temp_buffer_stateful = episodic_replay_buffer.StatefulEpisodicReplayBuffer(
          online_temp_buffer, num_episodes=num_eval_episodes)
        online_driver = safe_dynamic_episode_driver.SafeDynamicEpisodeDriver(
          sc_tf_env, online_collect_policy, online_temp_buffer, online_replay_buffer,
          observers=[online_temp_buffer_stateful.add_batch] + sc_metrics,
          num_episodes=num_eval_episodes)
      else:
        online_driver = online_driver_class(
          sc_tf_env, online_collect_policy, observers=[online_replay_buffer.add_batch] + sc_metrics,
          num_episodes=num_eval_episodes)
      # online_driver.run = common.function(online_driver.run, autograph=True)

    if eager_debug:
      tf.config.experimental_run_functions_eagerly(True)
    else:
      config_saver = gin.tf.GinConfigSaverHook(train_dir, summarize_config=True)
      tf.function(config_saver.after_create_session)()

    if agent_class not in SAFETY_AGENTS:
      collect_driver.run = common.function(collect_driver.run)

    if not rb_checkpointer.checkpoint_exists:  # and load_rb_ckpt_dir is None: # and pretraining:
      logging.info('Performing initial collection ...')
      init_collect_observers = agent_observers + train_metrics + env_metrics
      if agent_class in SAFETY_AGENTS:
        init_collect_observers += [online_replay_buffer.add_batch]
      initial_collect_driver_class(
          tf_env,
          initial_collect_policy,
          observers=init_collect_observers).run()
      last_id = replay_buffer._get_last_id()  # pylint: disable=protected-access
      logging.info('Data saved after initial collection: %d steps', last_id)
      if agent_class in SAFETY_AGENTS:
        last_id = online_replay_buffer._get_last_id()  # pylint: disable=protected-access
        logging.debug('Data saved in online buffer after initial collection: %d steps', last_id)

    if run_eval:
      results = metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='EvalMetrics',
      )
      if eval_metrics_callback is not None:
        eval_metrics_callback(results, global_step.numpy())
      metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
    iterator = iter(dataset)
    if agent_class in SAFETY_AGENTS:
      sc_dataset_pos = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=int(batch_size / 2), num_steps=2).prefetch(3)
      sc_dataset_neg = online_replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=int(batch_size / 2), num_steps=2).prefetch(3)
      # online_dataset = online_replay_buffer.as_dataset(
      #     num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
      sc_iter_pos = iter(sc_dataset_pos)
      sc_iter_neg = iter(sc_dataset_neg)
      dataset_spec = sc_dataset_neg.unbatch().element_spec[0]
      # online_iterator = iter(online_dataset)
      critic_metrics = [tf.keras.metrics.AUC(name='safety_critic_auc'),
                       tf.keras.metrics.TruePositives(name='safety_critic_tp', thresholds=[tf_agent._target_safety, 0.5]),
                       tf.keras.metrics.FalsePositives(name='safety_critic_fp', thresholds=[tf_agent._target_safety, 0.5]),
                       tf.keras.metrics.TrueNegatives(name='safety_critic_tn', thresholds=[tf_agent._target_safety, 0.5]),
                       tf.keras.metrics.FalseNegatives(name='safety_critic_fn', thresholds=[tf_agent._target_safety, 0.5]),
                       tf.keras.metrics.BinaryAccuracy(name='safety_critic_acc', threshold=tf_agent._target_safety)]
      # rb_rewards = None
      @common.function
      def critic_train_step():
        """Builds critic training step."""
        start_time = time.time()
        pos_experience, _ = next(sc_iter_pos)
        neg_experience, _ = next(sc_iter_neg)

        # experience, buf_info = next(online_iterator)
        experience = nest_utils.stack_nested_tensors(
          nest_utils.unstack_nested_tensors(pos_experience, dataset_spec)
          + nest_utils.unstack_nested_tensors(neg_experience, dataset_spec))
        boundary_mask = tf.logical_not(experience.is_boundary()[:, 0])
        experience = nest_utils.fast_map_structure(lambda *x: tf.boolean_mask(*x, boundary_mask), experience)
        if env_name.split('-')[0] in SAFETY_ENVS:
          safe_rew = experience.observation['task_agn_rew'][:, 1]
        else:
          assert False, "ENV: {} not in SAFETY_ENVS".format(env_name)
          # safe_rew = rb_rewards
          # safe_rew = tf.gather(safe_rew, tf.squeeze(buf_info.ids), axis=1)
        weights = (safe_rew / tf.reduce_mean(safe_rew+1e-16) + (1-safe_rew) / (tf.reduce_mean(1-safe_rew))) / 2
        # weights = 1 - safe_rew + safe_rew * (tf.reduce_mean(1-safe_rew) / 2 * tf.reduce_mean(safe_rew + 1e-16))
        ret = tf_agent.train_sc(experience, safe_rew, weights=weights, metrics=critic_metrics, training=updating_sc)
        logging.debug('critic train step: {} sec'.format(time.time() - start_time))
        return ret

    @common.function
    def train_step():
      experience, _ = next(iterator)
      ret = tf_agent.train(experience)
      return ret

    if not early_termination_fn:
      early_termination_fn = lambda: False

    loss_diverged = False
    # How many consecutive steps was loss diverged for.
    loss_divergence_counter = 0
    mean_train_loss = tf.keras.metrics.Mean(name='mean_train_loss')

    if online_critic:
      logging.debug('starting safety critic pretraining')
      safety_eps = tf_agent._safe_policy._safety_threshold
      tf_agent._safe_policy._safety_threshold = 0.6
      resample_counter = online_collect_policy._resample_counter
      mean_resample_ac = tf.keras.metrics.Mean(name='mean_unsafe_ac_freq')
      # don't fine-tune safety critic
      if (global_step.numpy() == 0 and load_root_dir is None):
        # rb_rewards = misc.process_replay_buffer(online_replay_buffer, as_tensor=True)
        for _ in range(train_sc_steps):
          sc_loss, lambda_loss = critic_train_step()  # pylint: disable=unused-variable
      tf_agent._safe_policy._safety_threshold = safety_eps
      critic_results = []
      for critic_metric in critic_metrics:
        critic_results.append((critic_metric.name, critic_metric.result().numpy()))
        critic_metric.reset_states()
      if train_metrics_callback:
        train_metrics_callback(collections.OrderedDict(critic_results), step=global_step.numpy())

    curr_ep = []

    logging.debug('starting policy training')
    while (global_step.numpy() <= num_global_steps and not early_termination_fn()):
      start_time = time.time()
      current_step = global_step.numpy()

      # MEASURE ACTION RESAMPLING FREQUENCY
      if online_critic:
        mean_resample_ac(resample_counter.result())
        resample_counter.reset()
        if time_step is None or time_step.is_last():
          resample_ac_freq = mean_resample_ac.result()
          mean_resample_ac.reset_states()

      # RUN COLLECTION
      time_step, policy_state = collect_driver.run(
          time_step=time_step,
          policy_state=policy_state,
      )

      # get last step taken
      traj = replay_buffer._data_table.read(replay_buffer._get_last_id() % replay_buffer._capacity)
      curr_ep.append(traj)

      if time_step.is_last():
        if (agent_class == safe_sac_agent.SafeSacAgentOnline
                and time_step.observation['task_agn_rew']):
          for traj in curr_ep:
            online_replay_buffer.add_batch(traj)
        curr_ep = []
        if agent_class == wcpg_agent.WcpgAgent:
          collect_policy._alpha = None  # reset WCPG alpha

      if current_step % log_interval == 0:
        logging.debug('policy eval: {} sec'.format(time.time() - start_time))

      # PERFORMS TRAIN STEP ON ALGORITHM (OFF-POLICY)
      train_time = time.time()
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
        mean_train_loss(train_loss.loss)
      if current_step == 0:
        logging.debug('train policy: {} sec'.format(time.time() - train_time))

      if train_metrics_callback is not None:
        train_metrics_callback(
          collections.OrderedDict([(k, v.numpy()) for k, v in train_loss.extra._asdict().items()]),
          step=global_step.numpy())
        train_metrics_callback(
          {'train_loss': mean_train_loss.result().numpy()}, step=global_step.numpy())

      # TRAIN (or evaluate) SAFETY CRITIC
      if agent_class in SAFETY_AGENTS and current_step % train_sc_interval == 0 and (pretraining or finetune_sc):
          batch_time_step = sc_tf_env.reset()
          batch_policy_state = online_collect_policy.get_initial_state(sc_tf_env.batch_size)
          if online_critic:
            online_driver.run(time_step=batch_time_step, policy_state=batch_policy_state)
          # rb_rewards = misc.process_replay_buffer(online_replay_buffer, as_tensor=True)
          for _ in range(train_sc_steps):  # epoch = train_sc_steps(10) * batch_size(256)
            sc_loss, lambda_loss = critic_train_step()  # pylint: disable=unused-variable

          critic_results = []
          for critic_metric in critic_metrics:
            critic_results.append((critic_metric.name, critic_metric.result().numpy()))
            critic_metric.reset_states()
          if train_metrics_callback is not None:
            train_metrics_callback(collections.OrderedDict(critic_results), step=global_step.numpy())

          sc_results = [('sc_loss', sc_loss.numpy()), ('lambda_loss', lambda_loss.numpy())]
          metric_utils.log_metrics(sc_metrics)
          with sc_summary_writer.as_default():
            for sc_metric in sc_metrics:
              sc_metric.tf_summaries(
                train_step=global_step, step_metrics=sc_metrics[:2])
              sc_results.append((sc_metric.name, sc_metric.result().numpy()))
            tf.compat.v2.summary.scalar(
              name='resample_ac_freq', data=resample_ac_freq, step=global_step)
          if train_metrics_callback:
            train_metrics_callback(collections.OrderedDict(sc_results), step=global_step.numpy())

      total_loss = mean_train_loss.result()
      mean_train_loss.reset_states()
      # Check for exploding losses.
      if (math.isnan(total_loss) or math.isinf(total_loss) or
          total_loss > MAX_LOSS):
        loss_divergence_counter += 1
        if loss_divergence_counter > TERMINATE_AFTER_DIVERGED_LOSS_STEPS:
          loss_diverged = True
          logging.debug('Loss diverged, critic_loss: %s, actor_loss: %s',
                        train_loss.extra.critic_loss, train_loss.extra.actor_loss)
          break
      else:
        loss_divergence_counter = 0

      time_acc += time.time() - start_time

      if current_step % log_interval == 0:
        metric_utils.log_metrics(train_metrics)
        logging.info('step = %d, loss = %f', current_step, total_loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      train_results = []
      for train_metric in train_metrics:
        if isinstance(train_metric, (metrics.AverageEarlyFailureMetric,
                                     metrics.AverageFallenMetric,
                                     metrics.AverageSuccessMetric)):
          # Plot failure as a fn of return
          train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:3])
        else:
          train_metric.tf_summaries(
              train_step=global_step, step_metrics=train_metrics[:2])
        train_results.append((train_metric.name, train_metric.result().numpy()))
      if env_metrics:
        for env_metric in env_metrics:
          env_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])
          train_results.append((env_metric.name, env_metric.result().numpy()))
      if train_metrics_callback is not None:
        train_metrics_callback(collections.OrderedDict(train_results), step=global_step.numpy())

      global_step_val = global_step.numpy()
      if global_step_val % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step_val)

      if global_step_val % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=global_step_val)
        if agent_class in SAFETY_AGENTS:
          safety_critic_checkpointer.save(global_step=global_step_val)

      if rb_checkpoint_interval and global_step_val % rb_checkpoint_interval == 0:
        if agent_class in SAFETY_AGENTS:
          online_rb_checkpointer.save(global_step=global_step_val)
        rb_checkpointer.save(global_step=global_step_val)
      # elif online_critic:
      #   clear_rb()
      if wandb and current_step % eval_interval == 0 and "Drunk" in env_name:
        misc.record_point_mass_episode(eval_tf_env, eval_policy, global_step_val)
        if online_critic:
          misc.record_point_mass_episode(eval_tf_env, tf_agent._safe_policy, global_step_val, 'safe-trajectory')

      if run_eval and global_step_val % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='EvalMetrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step_val)
        metric_utils.log_metrics(eval_metrics)
        eval_results = []
        for eval_metric in eval_metrics:
          if isinstance(eval_metric, (metrics.AverageEarlyFailureMetric,
                                       metrics.AverageFallenMetric,
                                       metrics.AverageSuccessMetric)):
            # Plot failure as a fn of return
            eval_metric.tf_summaries(
              train_step=global_step, step_metrics=eval_metrics[:3])
          else:
            eval_metric.tf_summaries(
              train_step=global_step, step_metrics=eval_metrics[:2])
          eval_results.append((eval_metric.name, eval_metric.result().numpy()))
        if env_metrics:
          for env_metric in env_metrics:
            env_metric.tf_summaries(
              train_step=global_step, step_metrics=eval_metrics[:2])
            eval_results.append((env_metric.name, env_metric.result().numpy()))
        if train_metrics_callback is not None:
          train_metrics_callback(collections.OrderedDict(eval_results), step=global_step.numpy())

      if monitor and current_step % monitor_interval == 0:
        monitor_time_step = monitor_py_env.reset()
        monitor_policy_state = eval_policy.get_initial_state(1)
        ep_len = 0
        monitor_start = time.time()
        while not monitor_time_step.is_last():
          monitor_action = eval_policy.action(monitor_time_step, monitor_policy_state)
          action, monitor_policy_state = monitor_action.action, monitor_action.state
          monitor_time_step = monitor_py_env.step(action)
          ep_len += 1
        logging.debug('saved rollout at timestep {}, rollout length: {}, {} sec'.format(
            global_step_val, ep_len, time.time() - monitor_start))

      if current_step % log_interval == 0:
        logging.debug('iteration time: {} sec'.format(time.time() - start_time))

  if not keep_rb_checkpoint:
    misc.cleanup_checkpoints(rb_ckpt_dir)

  if loss_diverged:
    # Raise an error at the very end after the cleanup.
    raise ValueError('Loss diverged to {} at step {}, terminating.'.format(
        total_loss, global_step.numpy()))

  return total_loss
