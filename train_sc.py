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

# Lint as: python3
"""Trains and evaluates Safety Critic offline.

Trains and evaluates safety critic on train and test replay buffers, and plots
AUC, Acc, and related metrics.
"""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import os
import os.path as osp
from datetime import datetime
import collections

from absl import logging
# from algos import safe_sac_agent
import gin
import time
import gin.tf
import gym
import tensorflow as tf

from tf_agents.agents.sac import sac_agent
from tf_agents.agents.tf_agent import LossInfo
from tf_agents.environments import tf_py_environment, gym_wrapper
from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics, tf_py_metric, py_metrics
from tf_agents.trajectories import trajectory
from tf_agents.utils import common, nest_utils
from safemrl.algos import agents
from safemrl.algos import safe_sac_agent
from safemrl.algos import ensemble_sac_agent
from safemrl.algos import wcpg_agent
from safemrl.utils import misc
from safemrl.utils import data_utils, train_utils, log_utils

SafetyCriticLossInfo = collections.namedtuple(
  "SafetyCriticLossInfo", ("loss", "extra"))

SAFETY_AGENTS = [safe_sac_agent.SqrlAgent]
ALGOS = {'sac': sac_agent.SacAgent,
         'sqrl': safe_sac_agent.SqrlAgent,
         'sac_ensemble': ensemble_sac_agent.EnsembleSacAgent,
         'wcpg': wcpg_agent.WcpgAgent}


def experience_to_transitions(experience):
  boundary_mask = tf.logical_not(experience.is_boundary()[:, 0])
  experience = nest_utils.fast_map_structure(
    lambda *x: tf.boolean_mask(*x, boundary_mask), experience)
  time_steps, policy_steps, next_time_steps = (
    trajectory.experience_to_transitions(experience, True))
  actions = policy_steps.action
  return time_steps, actions, next_time_steps


@gin.configurable(whitelist=['safety_gamma', 'loss_fn'])
def safety_critic_loss(time_steps,
                       actions,
                       next_time_steps,
                       safety_rewards,
                       get_action,
                       global_step,
                       critic_network=None,
                       target_network=None,
                       target_safety=None,
                       safety_gamma=0.45,
                       loss_fn='bce',
                       metrics=None,
                       debug_summaries=False):
  """Computes the critic loss for SAC training.

  Args:
    time_steps: A batch of timesteps.
    actions: A batch of actions.
    next_time_steps: A batch of next timesteps.
    safety_rewards: Task-agnostic rewards for safety. 1 is unsafe, 0 is safe.
    weights: Optional scalar or elementwise (per-batch-entry) importance
      weights.

  Returns:
    safe_critic_loss: A scalar critic loss.
  """
  with tf.name_scope('safety_critic_loss'):
    next_actions = get_action(next_time_steps)
    target_input = (next_time_steps.observation, next_actions)
    target_q_values, _ = target_network(
      target_input, next_time_steps.step_type)
    target_q_values = tf.nn.sigmoid(target_q_values)
    td_targets = tf.stop_gradient(
      safety_rewards + (1 - safety_rewards) * safety_gamma *
      next_time_steps.discount * target_q_values)

    if loss_fn == 'bce' or loss_fn == tf.keras.losses.binary_crossentropy:
      td_targets = tf.nn.sigmoid(td_targets)

    pred_input = (time_steps.observation, actions)
    pred_td_targets, _ = critic_network(pred_input, time_steps.step_type,
                                        training=True)
    pred_td_targets = tf.nn.sigmoid(pred_td_targets)

    # Loss fns: binary_crossentropy/squared_difference
    if loss_fn == 'mse':
      sc_loss = tf.math.squared_difference(td_targets, pred_td_targets)
    elif loss_fn == 'bce' or loss_fn is None:
      sc_loss = tf.keras.losses.binary_crossentropy(td_targets,
                                                    pred_td_targets)
    elif loss_fn is not None:
      sc_loss = loss_fn(td_targets, pred_td_targets)

    if metrics:
      for metric in metrics:
        if isinstance(metric, tf.keras.metrics.AUC):
          metric.update_state(safety_rewards, pred_td_targets)
        else:
          rew_pred = tf.greater_equal(pred_td_targets, target_safety)
          metric.update_state(safety_rewards, rew_pred)

    if debug_summaries:
      pred_td_targets = tf.nn.sigmoid(pred_td_targets)
      td_errors = td_targets - pred_td_targets
      common.generate_tensor_summaries('safety_td_errors', td_errors,
                                       global_step)
      common.generate_tensor_summaries('safety_td_targets', td_targets,
                                       global_step)
      common.generate_tensor_summaries('safety_pred_td_targets',
                                       pred_td_targets,
                                       global_step)

    return sc_loss

@gin.configurable(whitelist=['alpha', 'target_safety'])
@common.function
def train_step(exp, safe_rew, tf_agent, sc_net=None, target_sc_net=None,
               global_step=None, weights=None,
               target_update=None, metrics=None,
               optimizer=None, alpha=2., target_safety=None,
               debug_summaries=False):
  sc_net = sc_net or tf_agent._safety_critic_network
  target_sc_net = target_sc_net or tf_agent._target_safety_critic_network
  target_update = target_update or tf_agent._update_target_safety_critic
  optimizer = optimizer or tf_agent._safety_critic_optimizer
  get_action = lambda ts: tf_agent._actions_and_log_probs(ts)[0]

  time_steps, actions, next_time_steps = experience_to_transitions(exp)

  # update safety critic
  trainable_safety_variables = sc_net.trainable_variables
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    assert trainable_safety_variables, ('No trainable safety critic variables'
                                        ' to optimize.')
    tape.watch(trainable_safety_variables)
    sc_loss = safety_critic_loss(
      time_steps,
      actions,
      next_time_steps,
      safe_rew,
      get_action,
      global_step,
      critic_network=sc_net,
      target_network=target_sc_net,
      target_safety=target_safety,
      metrics=metrics,
      debug_summaries=debug_summaries)

    sc_loss_raw = tf.reduce_mean(sc_loss)

    if weights is not None:
      sc_loss *= weights

    # Take the mean across the batch.
    sc_loss = tf.reduce_mean(sc_loss)

    q_safe = train_utils.eval_safety(sc_net, get_action, time_steps)
    lam_loss = tf.reduce_mean(q_safe - tf_agent._target_safety)
    total_loss = sc_loss + alpha * lam_loss

    tf.debugging.check_numerics(sc_loss, 'Critic loss is inf or nan.')
    safety_critic_grads = tape.gradient(total_loss,
                                        trainable_safety_variables)
    tf_agent._apply_gradients(safety_critic_grads, trainable_safety_variables,
                              optimizer)

  # update target safety critic independently of target critic during pretraining
  target_update()

  return total_loss, sc_loss_raw, lam_loss


@gin.configurable(blacklist=['seed', 'monitor'])
def train_eval(load_root_dir,
               env_load_fn=None,
               gym_env_wrappers=[],
               monitor=False,
               env_name=None,
               agent_class=None,
               train_metrics_callback=None,
               # SacAgent args
               actor_fc_layers=(256, 256),
               critic_joint_fc_layers=(256, 256),
               # Safety Critic training args
               safety_critic_joint_fc_layers=None,
               safety_critic_lr=3e-4,
               safety_critic_bias_init_val=None,
               safety_critic_kernel_scale=None,
               n_envs=None,
               target_safety=0.2,
               fail_weight=None,
               # Params for train
               num_global_steps=10000,
               batch_size=256,
               # Params for eval
               run_eval=False,
               eval_metrics=[],
               num_eval_episodes=10,
               eval_interval=1000,
               # Params for summaries and logging
               train_checkpoint_interval=10000,
               summary_interval=1000,
               monitor_interval=5000,
               summaries_flush_secs=10,
               debug_summaries=False,
               seed=None):

  if isinstance(agent_class, str):
    assert agent_class in ALGOS, 'trainer.train_eval: agent_class {} invalid'.format(agent_class)
    agent_class = ALGOS.get(agent_class)

  train_ckpt_dir = osp.join(load_root_dir, 'train')
  rb_ckpt_dir = osp.join(load_root_dir, 'train', 'replay_buffer')

  py_env = env_load_fn(env_name, gym_env_wrappers=gym_env_wrappers)
  tf_env = tf_py_environment.TFPyEnvironment(py_env)

  if monitor:
    vid_path = os.path.join(load_root_dir, 'rollouts')
    monitor_env_wrapper = misc.monitor_freq(1, vid_path)
    monitor_env = gym.make(env_name)
    for wrapper in gym_env_wrappers:
      monitor_env = wrapper(monitor_env)
    monitor_env = monitor_env_wrapper(monitor_env)
    # auto_reset must be False to ensure Monitor works correctly
    monitor_py_env = gym_wrapper.GymWrapper(monitor_env, auto_reset=False)

  if run_eval:
    eval_dir = os.path.join(load_root_dir, 'eval')
    n_envs = n_envs or num_eval_episodes
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(prefix='EvalMetrics',
                                       buffer_size=num_eval_episodes,
                                       batch_size=n_envs),
        tf_metrics.AverageEpisodeLengthMetric(prefix='EvalMetrics',
                                              buffer_size=num_eval_episodes,
                                              batch_size=n_envs)
      ] + [tf_py_metric.TFPyMetric(m, name='EvalMetrics/{}'.format(m.name))
           for m in eval_metrics]
    eval_tf_env = tf_py_environment.TFPyEnvironment(
      parallel_py_environment.ParallelPyEnvironment(
        [lambda: env_load_fn(env_name, gym_env_wrappers=gym_env_wrappers)] * n_envs
      ))
    if seed:
      seeds = [seed * n_envs + i for i in range(n_envs)]
      try:
        eval_tf_env.pyenv.seed(seeds)
      except:
        pass

  global_step = tf.compat.v1.train.get_or_create_global_step()

  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()

  actor_net = actor_distribution_network.ActorDistributionNetwork(
    observation_spec,
    action_spec,
    fc_layer_params=actor_fc_layers,
    continuous_projection_net=agents.normal_projection_net)

  critic_net = agents.CriticNetwork(
    (observation_spec, action_spec),
    joint_fc_layer_params=critic_joint_fc_layers)

  if agent_class in SAFETY_AGENTS:
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
      debug_summaries=False)
  else:
    tf_agent = agent_class(
      time_step_spec,
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      train_step_counter=global_step,
      debug_summaries=False)

  collect_data_spec = tf_agent.collect_data_spec
  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      collect_data_spec,
      batch_size=1,
      max_length=1000000)
  replay_buffer = misc.load_rb_ckpt(rb_ckpt_dir, replay_buffer)

  tf_agent, _ = misc.load_agent_ckpt(train_ckpt_dir, tf_agent)
  if agent_class in SAFETY_AGENTS:
    target_safety = target_safety or tf_agent._target_safety
  loaded_train_steps = global_step.numpy()
  logging.info("Loaded agent from %s trained for %d steps", train_ckpt_dir,
               loaded_train_steps)
  global_step.assign(0)
  tf.summary.experimental.set_step(global_step)

  thresholds = [target_safety, 0.5]
  sc_metrics = [tf.keras.metrics.AUC(name='safety_critic_auc'),
                tf.keras.metrics.BinaryAccuracy(name='safety_critic_acc',
                                                threshold=0.5),
                tf.keras.metrics.TruePositives(name='safety_critic_tp',
                                               thresholds=thresholds),
                tf.keras.metrics.FalsePositives(name='safety_critic_fp',
                                                thresholds=thresholds),
                tf.keras.metrics.TrueNegatives(name='safety_critic_tn',
                                               thresholds=thresholds),
                tf.keras.metrics.FalseNegatives(name='safety_critic_fn',
                                                thresholds=thresholds)
                ]

  if seed:
    tf.compat.v1.set_random_seed(seed)

  summaries_flush_secs = 10
  timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
  offline_train_dir = osp.join(train_ckpt_dir, 'offline', timestamp)
  config_saver = gin.tf.GinConfigSaverHook(offline_train_dir,
                                           summarize_config=True)
  tf.function(config_saver.after_create_session)()

  sc_summary_writer = tf.compat.v2.summary.create_file_writer(
    offline_train_dir, flush_millis=summaries_flush_secs * 1000)
  sc_summary_writer.set_as_default()

  if safety_critic_kernel_scale is not None:
    ki = tf.compat.v1.variance_scaling_initializer(
      scale=safety_critic_kernel_scale, mode='fan_in',
      distribution='truncated_normal')
  else:
    ki = tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform')

  if safety_critic_bias_init_val is not None:
    bi = tf.constant_initializer(safety_critic_bias_init_val)
  else:
    bi = None
  sc_net_off = agents.CriticNetwork(
    (observation_spec, action_spec),
    joint_fc_layer_params=safety_critic_joint_fc_layers,
    kernel_initializer=ki,
    value_bias_initializer=bi,
    name='SafetyCriticOffline')
  sc_net_off.create_variables()
  target_sc_net_off = common.maybe_copy_target_network_with_checks(
    sc_net_off, None, 'TargetSafetyCriticNetwork')
  optimizer = tf.keras.optimizers.Adam(safety_critic_lr)
  sc_net_off_ckpt_dir = os.path.join(offline_train_dir, 'safety_critic')
  sc_checkpointer = common.Checkpointer(
      ckpt_dir=sc_net_off_ckpt_dir,
      safety_critic=sc_net_off,
      target_safety_critic=target_sc_net_off,
      optimizer=optimizer,
      global_step=global_step,
      max_to_keep=5)
  sc_checkpointer.initialize_or_restore()

  resample_counter = py_metrics.CounterMetric('ActionResampleCounter')
  eval_policy = agents.SafeActorPolicyRSVar(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=actor_net,
        safety_critic_network=sc_net_off,
        safety_threshold=target_safety,
        resample_counter=resample_counter,
        training=True)

  dataset = replay_buffer.as_dataset(num_parallel_calls=3, num_steps=2,
                                     sample_batch_size=batch_size//2).prefetch(3)
  data = iter(dataset)
  full_data = replay_buffer.gather_all()

  fail_mask = tf.cast(full_data.observation['task_agn_rew'], tf.bool)
  fail_step = nest_utils.fast_map_structure(
    lambda *x: tf.boolean_mask(*x, fail_mask), full_data)
  init_step = nest_utils.fast_map_structure(
    lambda *x: tf.boolean_mask(*x, full_data.is_first()), full_data)
  before_fail_mask = tf.roll(fail_mask, [-1], axis=[1])
  after_init_mask = tf.roll(full_data.is_first(), [1], axis=[1])
  before_fail_step = nest_utils.fast_map_structure(
    lambda *x: tf.boolean_mask(*x, before_fail_mask), full_data)
  after_init_step = nest_utils.fast_map_structure(
    lambda *x: tf.boolean_mask(*x, after_init_mask), full_data)

  filter_mask = tf.squeeze(tf.logical_or(before_fail_mask, fail_mask))
  filter_mask = tf.pad(filter_mask, [[0, replay_buffer._max_length - filter_mask.shape[0]]])
  n_failures = tf.reduce_sum(tf.cast(filter_mask, tf.int32)).numpy()

  failure_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    collect_data_spec, batch_size=1, max_length=n_failures,
    dataset_window_shift=1)
  data_utils.copy_rb(replay_buffer, failure_buffer, filter_mask)

  sc_dataset_neg = failure_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size // 2,
    num_steps=2).prefetch(3)
  neg_data = iter(sc_dataset_neg)

  get_action = lambda ts: tf_agent._actions_and_log_probs(ts)[0]
  eval_sc = log_utils.eval_fn(before_fail_step, fail_step, init_step,
                              after_init_step, get_action)

  losses = []
  mean_loss = tf.keras.metrics.Mean(name='mean_ep_loss')
  target_update = train_utils.get_target_updater(sc_net_off, target_sc_net_off)

  with tf.summary.record_if(
          lambda: tf.math.equal(global_step % summary_interval, 0)):
    while global_step.numpy() < num_global_steps:
      pos_experience, _ = next(data)
      neg_experience, _ = next(neg_data)
      exp = data_utils.concat_batches(pos_experience, neg_experience,
                                      collect_data_spec)
      boundary_mask = tf.logical_not(exp.is_boundary()[:, 0])
      exp = nest_utils.fast_map_structure(
        lambda *x: tf.boolean_mask(*x, boundary_mask), exp)
      safe_rew = exp.observation['task_agn_rew'][:, 1]
      if fail_weight:
        weights = tf.where(tf.cast(safe_rew, tf.bool),
                           fail_weight / 0.5, (1 - fail_weight) / 0.5)
      else:
        weights = None
      train_loss, sc_loss, lam_loss = train_step(exp, safe_rew, tf_agent,
                                                 sc_net=sc_net_off,
                                                 target_sc_net=target_sc_net_off,
                                                 metrics=sc_metrics,
                                                 weights=weights,
                                                 target_safety=target_safety,
                                                 optimizer=optimizer,
                                                 target_update=target_update,
                                                 debug_summaries=debug_summaries)
      global_step.assign_add(1)
      global_step_val = global_step.numpy()
      losses.append((train_loss.numpy(), sc_loss.numpy(), lam_loss.numpy()))
      mean_loss(train_loss)
      with tf.name_scope('Losses'):
        tf.compat.v2.summary.scalar(name='sc_loss', data=sc_loss,
                                    step=global_step_val)
        tf.compat.v2.summary.scalar(name='lam_loss', data=lam_loss,
                                    step=global_step_val)
        if global_step_val % summary_interval == 0:
          tf.compat.v2.summary.scalar(name=mean_loss.name,
                                      data=mean_loss.result(),
                                      step=global_step_val)
      if global_step_val % summary_interval == 0:
        with tf.name_scope('Metrics'):
          for metric in sc_metrics:
            if len(tf.squeeze(metric.result()).shape) == 0:
              tf.compat.v2.summary.scalar(name=metric.name, data=metric.result(),
                                          step=global_step_val)
            else:
              fmt_str = '_{}'.format(thresholds[0])
              tf.compat.v2.summary.scalar(name=metric.name + fmt_str,
                                          data=metric.result()[0],
                                          step=global_step_val)
              fmt_str = '_{}'.format(thresholds[1])
              tf.compat.v2.summary.scalar(name=metric.name + fmt_str,
                                          data=metric.result()[1],
                                          step=global_step_val)
            metric.reset_states()
      if global_step_val % eval_interval == 0:
        eval_sc(sc_net_off, step=global_step_val)
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
            train_metrics_callback(results, global_step_val)
          metric_utils.log_metrics(eval_metrics)
          with eval_summary_writer.as_default():
            for eval_metric in eval_metrics[2:]:
              eval_metric.tf_summaries(train_step=global_step,
                                       step_metrics=eval_metrics[:2])
      if monitor and global_step_val % monitor_interval == 0:
        monitor_time_step = monitor_py_env.reset()
        monitor_policy_state = eval_policy.get_initial_state(1)
        ep_len = 0
        monitor_start = time.time()
        while not monitor_time_step.is_last():
          monitor_action = eval_policy.action(monitor_time_step,
                                              monitor_policy_state)
          action, monitor_policy_state = monitor_action.action, monitor_action.state
          monitor_time_step = monitor_py_env.step(action)
          ep_len += 1
        logging.debug(
          'saved rollout at timestep %d, rollout length: %d, %4.2f sec',
          global_step_val, ep_len, time.time() - monitor_start)

      if global_step_val % train_checkpoint_interval == 0:
        sc_checkpointer.save(global_step=global_step_val)
