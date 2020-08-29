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
import gin
import gym
import robel
import tensorflow as tf

from absl import logging
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
from tf_agents.specs import tensor_spec
from tf_agents.utils import common, nest_utils

from safemrl.algos import agents
from safemrl.algos import safe_sac_agent
from safemrl.algos import ensemble_sac_agent
from safemrl.algos import wcpg_agent
from safemrl.utils import misc
from safemrl.utils import metrics
from safemrl.utils import train_utils

try:
  import highway_env
except ImportError:
  logging.debug("Could not import highway_env")

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9
SAFETY_ENVS = ['IndianWell', 'IndianWell2', 'IndianWell3', 'DrunkSpider', 'DrunkSpiderShort',
               'SafemrlCube', 'highway', 'MinitaurGoalVelocityEnv', 'MinitaurRandFrictionGoalVelocityEnv']
SAFETY_AGENTS = [safe_sac_agent.SqrlAgent]
ALGOS = {'sac': sac_agent.SacAgent,
         'sqrl': safe_sac_agent.SqrlAgent,
         'sac_ensemble': ensemble_sac_agent.EnsembleSacAgent,
         'wcpg': wcpg_agent.WcpgAgent}

# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_LOSS_STEPS = 100


@gin.configurable(blacklist=['seed', 'eager_debug', 'monitor'])
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
    rb_size=None,
    train_steps_per_iteration=1,
    train_metrics=None,
    eval_metrics=None,
    train_metrics_callback=None,
    # SacAgent args
    actor_fc_layers=(256, 256),
    critic_joint_fc_layers=(256, 256),
    # Safety Critic training args
    sc_rb_size=None,
    target_safety=None,
    train_sc_steps=10,
    train_sc_interval=1000,
    online_critic=False,
    n_envs=None,
    finetune_sc=False,
    pretraining=True,
    lambda_schedule_nsteps=0,
    lambda_initial=0.,
    lambda_final=1.,
    kstep_fail=0,
    # Ensemble Critic training args
    num_critics=None,
    critic_learning_rate=3e-4,
    # Wcpg Critic args
    critic_preprocessing_layer_size=256,
    # Params for train
    batch_size=256,
    # Params for eval
    run_eval=False,
    num_eval_episodes=10,
    eval_interval=1000,
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

  """train and eval script for SQRL."""
  if isinstance(agent_class, str):
    assert agent_class in ALGOS, 'trainer.train_eval: agent_class {} invalid'.format(agent_class)
    agent_class = ALGOS.get(agent_class)
  n_envs = n_envs or num_eval_episodes
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')

  # =====================================================================#
  #  Setup summary metrics, file writers, and create env                 #
  # =====================================================================#
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
    train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  train_metrics = train_metrics or []
  eval_metrics = eval_metrics or []

  updating_sc = online_critic and (not load_root_dir or finetune_sc)
  logging.debug('updating safety critic: %s', updating_sc)

  if seed:
    tf.compat.v1.set_random_seed(seed)

  if agent_class in SAFETY_AGENTS:
    if online_critic:
      sc_tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
          [lambda: env_load_fn(env_name)] * n_envs
        ))
      if seed:
        seeds = [seed * n_envs + i for i in range(n_envs)]
        try:
          sc_tf_env.pyenv.seed(seeds)
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
        [lambda: env_load_fn(env_name)] * n_envs
      ))
    if seed:
      try:
        for i, pyenv in enumerate(eval_tf_env.pyenv.envs):
          pyenv.seed(seed * n_envs + i)
      except:
        pass
  elif 'Drunk' in env_name:
    # Just visualizes trajectories in drunk spider environment
    eval_tf_env = tf_py_environment.TFPyEnvironment(
      env_load_fn(env_name))
  else:
    eval_tf_env = None

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

  with tf.summary.record_if(
          lambda: tf.math.equal(global_step % summary_interval, 0)):
    py_env = env_load_fn(env_name)
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

    # =====================================================================#
    #  Setup agent class                                                   #
    # =====================================================================#

    if agent_class == wcpg_agent.WcpgAgent:
      alpha_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, minimum=0., maximum=1.,
                                                 name='alpha')
      input_tensor_spec = (observation_spec, action_spec, alpha_spec)
      critic_net = agents.DistributionalCriticNetwork(
        input_tensor_spec, preprocessing_layer_size=critic_preprocessing_layer_size,
        joint_fc_layer_params=critic_joint_fc_layers)
      actor_net = agents.WcpgActorNetwork((observation_spec, alpha_spec), action_spec)
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
      logging.debug('Making SQRL agent')
      if lambda_schedule_nsteps > 0:
        lambda_update_every_nsteps = num_global_steps // lambda_schedule_nsteps
        step_size = (lambda_final - lambda_initial) / lambda_update_every_nsteps
        lambda_scheduler = lambda lam: common.periodically(
          body=lambda: tf.group(lam.assign(lam + step_size)),
          period=lambda_update_every_nsteps)
      else:
        lambda_scheduler = None
      safety_critic_net = agents.CriticNetwork(
        (observation_spec, action_spec),
        joint_fc_layer_params=critic_joint_fc_layers)
      ts = target_safety
      thresholds = [ts, 0.5]
      sc_metrics = [tf.keras.metrics.AUC(name='safety_critic_auc'),
                    tf.keras.metrics.TruePositives(name='safety_critic_tp',
                                                   thresholds=thresholds),
                    tf.keras.metrics.FalsePositives(name='safety_critic_fp',
                                                    thresholds=thresholds),
                    tf.keras.metrics.TrueNegatives(name='safety_critic_tn',
                                                   thresholds=thresholds),
                    tf.keras.metrics.FalseNegatives(name='safety_critic_fn',
                                                    thresholds=thresholds),
                    tf.keras.metrics.BinaryAccuracy(name='safety_critic_acc',
                                                    threshold=0.5)]
      tf_agent = agent_class(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        safety_critic_network=safety_critic_net,
        train_step_counter=global_step,
        debug_summaries=debug_summaries,
        safety_pretraining=pretraining,
        train_critic_online=online_critic,
        initial_log_lambda=lambda_initial,
        log_lambda=(lambda_scheduler is None),
        lambda_scheduler=lambda_scheduler)
    elif agent_class is ensemble_sac_agent.EnsembleSacAgent:
      critic_nets, critic_optimizers = [critic_net], [tf.keras.optimizers.Adam(critic_learning_rate)]
      for _ in range(num_critics - 1):
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
    else:  # agent is either SacAgent or WcpgAgent
      logging.debug('critic input_tensor_spec: %s', critic_net.input_tensor_spec)
      tf_agent = agent_class(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        train_step_counter=global_step,
        debug_summaries=debug_summaries)

    tf_agent.initialize()

    # =====================================================================#
    #  Setup replay buffer                                                 #
    # =====================================================================#
    collect_data_spec = tf_agent.collect_data_spec

    logging.debug('Allocating replay buffer ...')
    # Add to replay buffer and other agent specific observers.
    rb_size = rb_size or 1000000
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      collect_data_spec,
      batch_size=1,
      max_length=rb_size)

    logging.debug('RB capacity: %i', replay_buffer.capacity)
    logging.debug('ReplayBuffer Collect data spec: %s', collect_data_spec)

    if agent_class in SAFETY_AGENTS:
      sc_rb_size = sc_rb_size or num_eval_episodes * 500
      sc_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        collect_data_spec, batch_size=1, max_length=sc_rb_size,
        dataset_window_shift=1)

    num_episodes = tf_metrics.NumberOfEpisodes()
    num_env_steps = tf_metrics.EnvironmentSteps()
    return_metric = tf_metrics.AverageReturnMetric(
      buffer_size=num_eval_episodes, batch_size=tf_env.batch_size)
    train_metrics = [
                      num_episodes, num_env_steps,
                      return_metric,
                      tf_metrics.AverageEpisodeLengthMetric(
                        buffer_size=num_eval_episodes, batch_size=tf_env.batch_size),
                    ] + [tf_py_metric.TFPyMetric(m) for m in train_metrics]

    if 'Minitaur' in env_name and not pretraining:
      goal_vel = gin.query_parameter("%GOAL_VELOCITY")
      early_termination_fn = train_utils.MinitaurTerminationFn(
        speed_metric=train_metrics[-2], total_falls_metric=train_metrics[-3],
        env_steps_metric=num_env_steps, goal_speed=goal_vel)

    if env_metric_factories:
      for env_metric in env_metric_factories:
        train_metrics.append(tf_py_metric.TFPyMetric(env_metric(tf_env.pyenv.envs)))
        if run_eval:
          eval_metrics.append(env_metric([env for env in
                                          eval_tf_env.pyenv._envs]))

    # =====================================================================#
    #  Setup collect policies                                              #
    # =====================================================================#
    if not online_critic:
      eval_policy = tf_agent.policy
      collect_policy = tf_agent.collect_policy
      if not pretraining and agent_class in SAFETY_AGENTS:
        collect_policy = tf_agent.safe_policy
    else:
      eval_policy = tf_agent.collect_policy if pretraining else tf_agent.safe_policy
      collect_policy = tf_agent.collect_policy if pretraining else tf_agent.safe_policy
      online_collect_policy = tf_agent.safe_policy  # if pretraining else tf_agent.collect_policy
      if pretraining:
        online_collect_policy._training = False

    if not load_root_dir:
      initial_collect_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec)
    else:
      initial_collect_policy = collect_policy
    if agent_class == wcpg_agent.WcpgAgent:
      initial_collect_policy = agents.WcpgPolicyWrapper(initial_collect_policy)

    # =====================================================================#
    #  Setup Checkpointing                                                 #
    # =====================================================================#
    train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      agent=tf_agent,
      global_step=global_step,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      policy=eval_policy,
      global_step=global_step)

    rb_ckpt_dir = os.path.join(train_dir, 'replay_buffer')
    rb_checkpointer = common.Checkpointer(
      ckpt_dir=rb_ckpt_dir, max_to_keep=1, replay_buffer=replay_buffer)

    if online_critic:
      online_rb_ckpt_dir = os.path.join(train_dir, 'online_replay_buffer')
      online_rb_checkpointer = common.Checkpointer(
        ckpt_dir=online_rb_ckpt_dir,
        max_to_keep=1,
        replay_buffer=sc_buffer)

    # loads agent, replay buffer, and online sc/buffer if online_critic
    if load_root_dir:
      load_root_dir = os.path.expanduser(load_root_dir)
      load_train_dir = os.path.join(load_root_dir, 'train')
      misc.load_agent_ckpt(load_train_dir, tf_agent)
      if len(os.listdir(os.path.join(load_train_dir, 'replay_buffer'))) > 1:
        load_rb_ckpt_dir = os.path.join(load_train_dir, 'replay_buffer')
        misc.load_rb_ckpt(load_rb_ckpt_dir, replay_buffer)
      if online_critic:
        load_online_sc_ckpt_dir = os.path.join(load_root_dir, 'sc')
        load_online_rb_ckpt_dir = os.path.join(load_train_dir,
                                               'online_replay_buffer')
        if osp.exists(load_online_rb_ckpt_dir):
          misc.load_rb_ckpt(load_online_rb_ckpt_dir, sc_buffer)
        if osp.exists(load_online_sc_ckpt_dir):
          misc.load_safety_critic_ckpt(load_online_sc_ckpt_dir,
                                       safety_critic_net)
      elif agent_class in SAFETY_AGENTS:
        offline_run = sorted(os.listdir(os.path.join(load_train_dir, 'offline')))[-1]
        load_sc_ckpt_dir = os.path.join(load_train_dir, 'offline',
                                        offline_run, 'safety_critic')
        if osp.exists(load_sc_ckpt_dir):
          sc_net_off = agents.CriticNetwork(
            (observation_spec, action_spec),
            joint_fc_layer_params=(512, 512),
            name='SafetyCriticOffline')
          sc_net_off.create_variables()
          target_sc_net_off = common.maybe_copy_target_network_with_checks(
            sc_net_off, None, 'TargetSafetyCriticNetwork')
          sc_optimizer = tf.keras.optimizers.Adam(critic_learning_rate)
          _ = misc.load_safety_critic_ckpt(
            load_sc_ckpt_dir, safety_critic_net=sc_net_off,
            target_safety_critic=target_sc_net_off,
            optimizer=sc_optimizer)
          tf_agent._safety_critic_network = sc_net_off
          tf_agent._target_safety_critic_network = target_sc_net_off
          tf_agent._safety_critic_optimizer = sc_optimizer
    else:
      train_checkpointer.initialize_or_restore()
      rb_checkpointer.initialize_or_restore()
      if online_critic:
        online_rb_checkpointer.initialize_or_restore()

    if agent_class in SAFETY_AGENTS:
      sc_dir = os.path.join(root_dir, 'sc')
      safety_critic_checkpointer = common.Checkpointer(
        ckpt_dir=sc_dir,
        safety_critic=tf_agent._safety_critic_network,
        # pylint: disable=protected-access
        target_safety_critic=tf_agent._target_safety_critic_network,
        optimizer=tf_agent._safety_critic_optimizer,
        global_step=global_step)

      if not (load_root_dir and not online_critic):
        safety_critic_checkpointer.initialize_or_restore()

    agent_observers = [replay_buffer.add_batch] + train_metrics
    collect_driver = collect_driver_class(
      tf_env, collect_policy, observers=agent_observers)
    collect_driver.run = common.function_in_tf1()(collect_driver.run)

    if online_critic:
      logging.debug('online driver class: %s', online_driver_class)
      online_agent_observers = [num_episodes, num_env_steps,
                                sc_buffer.add_batch]
      online_driver = online_driver_class(
        sc_tf_env, online_collect_policy, observers=online_agent_observers,
        num_episodes=num_eval_episodes)
      online_driver.run = common.function_in_tf1()(online_driver.run)

    if eager_debug:
      tf.config.experimental_run_functions_eagerly(True)
    else:
      config_saver = gin.tf.GinConfigSaverHook(train_dir, summarize_config=True)
      tf.function(config_saver.after_create_session)()

    if global_step == 0:
      logging.info('Performing initial collection ...')
      init_collect_observers = agent_observers
      if agent_class in SAFETY_AGENTS:
        init_collect_observers += [sc_buffer.add_batch]
      initial_collect_driver_class(
        tf_env,
        initial_collect_policy,
        observers=init_collect_observers).run()
      last_id = replay_buffer._get_last_id()  # pylint: disable=protected-access
      logging.info('Data saved after initial collection: %d steps', last_id)
      if agent_class in SAFETY_AGENTS:
        last_id = sc_buffer._get_last_id()  # pylint: disable=protected-access
        logging.debug('Data saved in sc_buffer after initial collection: %d steps', last_id)

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
      if train_metrics_callback is not None:
        train_metrics_callback(results, global_step.numpy())
      metric_utils.log_metrics(eval_metrics)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    train_step = train_utils.get_train_step(tf_agent, replay_buffer, batch_size)

    if agent_class in SAFETY_AGENTS:
      critic_train_step = train_utils.get_critic_train_step(
        tf_agent, replay_buffer, sc_buffer, batch_size=batch_size,
        updating_sc=updating_sc, metrics=sc_metrics)

    if early_termination_fn is None:
      early_termination_fn = lambda: False

    loss_diverged = False
    # How many consecutive steps was loss diverged for.
    loss_divergence_counter = 0
    mean_train_loss = tf.keras.metrics.Mean(name='mean_train_loss')

    if agent_class in SAFETY_AGENTS:
      resample_counter = collect_policy._resample_counter
      mean_resample_ac = tf.keras.metrics.Mean(name='mean_unsafe_ac_freq')
      sc_metrics.append(mean_resample_ac)

      if online_critic:
        logging.debug('starting safety critic pretraining')
        # don't fine-tune safety critic
        if global_step.numpy() == 0:
          for _ in range(train_sc_steps):
            sc_loss, lambda_loss = critic_train_step()
          critic_results = [('sc_loss', sc_loss.numpy()), ('lambda_loss', lambda_loss.numpy())]
          for critic_metric in sc_metrics:
            res = critic_metric.result().numpy()
            if not res.shape:
              critic_results.append((critic_metric.name, res))
            else:
              for r, thresh in zip(res, thresholds):
                name = '_'.join([critic_metric.name, str(thresh)])
                critic_results.append((name, r))
            critic_metric.reset_states()
          if train_metrics_callback:
            train_metrics_callback(collections.OrderedDict(critic_results),
                                   step=global_step.numpy())

    logging.debug('Starting main train loop...')
    curr_ep = []
    global_step_val = global_step.numpy()
    while global_step_val <= num_global_steps and not early_termination_fn():
      start_time = time.time()

      # MEASURE ACTION RESAMPLING FREQUENCY
      if agent_class in SAFETY_AGENTS:
        if pretraining and global_step_val == num_global_steps // 2:
          if online_critic:
            online_collect_policy._training = True
          collect_policy._training = True
        if online_critic or collect_policy._training:
          mean_resample_ac(resample_counter.result())
          resample_counter.reset()
          if time_step is None or time_step.is_last():
            resample_ac_freq = mean_resample_ac.result()
            mean_resample_ac.reset_states()
            tf.compat.v2.summary.scalar(
              name='resample_ac_freq', data=resample_ac_freq, step=global_step)

      # RUN COLLECTION
      time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
      )

      # get last step taken by step_driver
      traj = replay_buffer._data_table.read(replay_buffer._get_last_id() %
                                            replay_buffer._capacity)
      curr_ep.append(traj)

      if time_step.is_last():
        if agent_class in SAFETY_AGENTS:
          if time_step.observation['task_agn_rew']:
            if kstep_fail:
              # applies task agn rew. over last k steps
              for i, traj in enumerate(curr_ep[-kstep_fail:]):
                traj.observation['task_agn_rew'] = 1.
                sc_buffer.add_batch(traj)
            else:
              [sc_buffer.add_batch(traj) for traj in curr_ep]
        curr_ep = []
        if agent_class == wcpg_agent.WcpgAgent:
          collect_policy._alpha = None  # reset WCPG alpha

      if (global_step_val + 1) % log_interval == 0:
        logging.debug('policy eval: %4.2f sec', time.time() - start_time)

      # PERFORMS TRAIN STEP ON ALGORITHM (OFF-POLICY)
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()
        mean_train_loss(train_loss.loss)

      current_step = global_step.numpy()
      total_loss = mean_train_loss.result()
      mean_train_loss.reset_states()

      if train_metrics_callback and current_step % summary_interval == 0:
        train_metrics_callback(
          collections.OrderedDict([(k, v.numpy()) for k, v in
                                   train_loss.extra._asdict().items()]),
          step=current_step)
        train_metrics_callback(
          {'train_loss': total_loss.numpy()}, step=current_step)

      # TRAIN AND/OR EVAL SAFETY CRITIC
      if agent_class in SAFETY_AGENTS and current_step % train_sc_interval == 0:
        if online_critic:
          batch_time_step = sc_tf_env.reset()

          # run online critic training collect & update
          batch_policy_state = online_collect_policy.get_initial_state(
            sc_tf_env.batch_size)
          online_driver.run(time_step=batch_time_step,
                            policy_state=batch_policy_state)
        for _ in range(train_sc_steps):
          sc_loss, lambda_loss = critic_train_step()
        # log safety_critic loss results
        critic_results = [('sc_loss', sc_loss.numpy()),
                          ('lambda_loss', lambda_loss.numpy())]
        metric_utils.log_metrics(sc_metrics)
        for critic_metric in sc_metrics:
          res = critic_metric.result().numpy()
          if not res.shape:
            critic_results.append((critic_metric.name, res))
          else:
            for r, thresh in zip(res, thresholds):
              name = '_'.join([critic_metric.name, str(thresh)])
              critic_results.append((name, r))
          critic_metric.reset_states()
        if train_metrics_callback and current_step % summary_interval == 0:
          train_metrics_callback(collections.OrderedDict(critic_results),
                                 step=current_step)

      # Check for exploding losses.
      if (math.isnan(total_loss) or math.isinf(total_loss) or
              total_loss > MAX_LOSS):
        loss_divergence_counter += 1
        if loss_divergence_counter > TERMINATE_AFTER_DIVERGED_LOSS_STEPS:
          loss_diverged = True
          logging.info('Loss diverged, critic_loss: %s, actor_loss: %s',
                       train_loss.extra.critic_loss,
                       train_loss.extra.actor_loss)
          break
      else:
        loss_divergence_counter = 0

      time_acc += time.time() - start_time

      # LOGGING AND METRICS
      if current_step % log_interval == 0:
        metric_utils.log_metrics(train_metrics)
        logging.info('step = %d, loss = %f', current_step, total_loss)
        steps_per_sec = (current_step - timed_at_step) / time_acc
        logging.info('%4.2f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
          name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = current_step
        time_acc = 0

      train_results = []

      for metric in train_metrics[2:]:
        if isinstance(metric, (metrics.AverageEarlyFailureMetric,
                               metrics.AverageFallenMetric,
                               metrics.AverageSuccessMetric)):
          # Plot failure as a fn of return
          metric.tf_summaries(
            train_step=global_step, step_metrics=[num_env_steps, num_episodes,
                                                  return_metric])
        else:
          metric.tf_summaries(
            train_step=global_step, step_metrics=[num_env_steps, num_env_steps])
        train_results.append((metric.name, metric.result().numpy()))

      if train_metrics_callback and current_step % summary_interval == 0:
        train_metrics_callback(collections.OrderedDict(train_results),
                               step=global_step.numpy())

      if current_step % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=current_step)

      if current_step % policy_checkpoint_interval == 0:
        policy_checkpointer.save(global_step=current_step)
        if agent_class in SAFETY_AGENTS:
          safety_critic_checkpointer.save(global_step=current_step)
          if online_critic:
            online_rb_checkpointer.save(global_step=current_step)

      if rb_checkpoint_interval and current_step % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=current_step)

      if wandb and current_step % eval_interval == 0 and "Drunk" in env_name:
        misc.record_point_mass_episode(eval_tf_env, eval_policy, current_step)
        if online_critic:
          misc.record_point_mass_episode(eval_tf_env, tf_agent.safe_policy,
                                         current_step, 'safe-trajectory')

      if run_eval and current_step % eval_interval == 0:
        eval_results = metric_utils.eager_compute(
          eval_metrics,
          eval_tf_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          train_step=global_step,
          summary_writer=eval_summary_writer,
          summary_prefix='EvalMetrics',
        )
        if train_metrics_callback is not None:
          train_metrics_callback(eval_results, current_step)
        metric_utils.log_metrics(eval_metrics)

        with eval_summary_writer.as_default():
          for eval_metric in eval_metrics[2:]:
            eval_metric.tf_summaries(train_step=global_step,
                                     step_metrics=eval_metrics[:2])

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
        logging.debug('saved rollout at timestep %d, rollout length: %d, %4.2f sec',
                      current_step, ep_len, time.time() - monitor_start)

      global_step_val = current_step

  if early_termination_fn():
    #  Early stopped, save all checkpoints if not saved
    if global_step_val % train_checkpoint_interval != 0:
      train_checkpointer.save(global_step=global_step_val)

    if global_step_val % policy_checkpoint_interval != 0:
      policy_checkpointer.save(global_step=global_step_val)
      if agent_class in SAFETY_AGENTS:
        safety_critic_checkpointer.save(global_step=global_step_val)
        if online_critic:
          online_rb_checkpointer.save(global_step=global_step_val)

    if rb_checkpoint_interval and global_step_val % rb_checkpoint_interval == 0:
      rb_checkpointer.save(global_step=global_step_val)

  if not keep_rb_checkpoint:
    misc.cleanup_checkpoints(rb_ckpt_dir)

  if loss_diverged:
    # Raise an error at the very end after the cleanup.
    raise ValueError('Loss diverged to {} at step {}, terminating.'.format(
      total_loss, global_step.numpy()))

  return total_loss
