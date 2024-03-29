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

"""Fork of TF-Agents SAC implementation with safety constaints.

Implementation of SAC using TF-Agents library, training a safety-critic and
using a safety policy that rejects unsafe actions.
SqrlAgent samples actions during pretraining with the safety critic.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import gin
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.policies import actor_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils

from safemrl.algos import agents

SafeSacLossInfo = collections.namedtuple(
    'SafeSacLossInfo',
    ('critic_loss', 'safety_critic_loss', 'actor_loss', 'alpha_loss', 'lambda_loss'))


@gin.configurable
def std_clip_transform(stddevs):
  stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 1),
                                  stddevs)
  return tf.exp(stddevs)


@gin.configurable
class SqrlAgent(sac_agent.SacAgent):
  """A Safe SAC Agent with 'online' safety critic updates."""

  def __init__(self,
               time_step_spec,
               action_spec,
               critic_network,
               actor_network,
               actor_optimizer,
               critic_optimizer,
               safety_critic_optimizer,
               alpha_optimizer,
               lambda_optimizer=None,
               lambda_scheduler=None,
               actor_policy_ctor=actor_policy.ActorPolicy,
               safety_critic_network=None,
               target_safety_critic_network=None,
               critic_network_2=None,
               target_critic_network=None,
               target_critic_network_2=None,
               target_update_tau=0.005,
               target_update_period=1,
               td_errors_loss_fn=tf.math.squared_difference,
               safe_td_errors_loss_fn=tf.keras.losses.binary_crossentropy,
               gamma=1.0,
               safety_gamma=None,
               reward_scale_factor=1.0,
               initial_log_alpha=0.0,
               initial_log_lambda=0.0,
               log_lambda=True,
               target_entropy=None,
               target_safety=0.1,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               safety_pretraining=True,
               train_critic_online=True,
               resample_counter=None,
               fail_weight=None,
               name="SqrlAgent"):
    self._safety_critic_network = safety_critic_network
    self._train_critic_online = train_critic_online
    super(SqrlAgent, self).__init__(
        time_step_spec=time_step_spec, action_spec=action_spec, critic_network=critic_network,
        actor_network=actor_network, actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer, critic_network_2=critic_network_2,
        target_critic_network=target_critic_network,
        target_critic_network_2=target_critic_network_2,
        actor_policy_ctor=actor_policy_ctor,
        target_update_tau=target_update_tau, target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn, gamma=gamma, reward_scale_factor=reward_scale_factor,
        initial_log_alpha=initial_log_alpha, target_entropy=target_entropy,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries, summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter, name=name)

    if safety_critic_network is not None:
      self._safety_critic_network.create_variables()
    else:
      self._safety_critic_network = common.maybe_copy_target_network_with_checks(
        critic_network, None, 'SafetyCriticNetwork')

    if target_safety_critic_network is not None:
      self._target_safety_critic_network = target_safety_critic_network
      self._target_safety_critic_network.create_variables()
    else:
      self._target_safety_critic_network = common.maybe_copy_target_network_with_checks(
        self._safety_critic_network, None, 'TargetSafetyCriticNetwork')

    lambda_name = 'initial_log_lambda' if log_lambda else 'initial_lambda'
    self._lambda_var = common.create_variable(
        lambda_name,
        initial_value=initial_log_lambda,
        dtype=tf.float32,
        trainable=True)
    self._target_safety = target_safety
    self._safe_policy = agents.SafeActorPolicyRSVar(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        safety_critic_network=self._safety_critic_network,
        safety_threshold=target_safety,
        resample_counter=resample_counter,
        training=True)
    self._collect_policy = agents.SafeActorPolicyRSVar(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        actor_network=self._actor_network,
        safety_critic_network=self._safety_critic_network,
        safety_threshold=target_safety,
        resample_counter=resample_counter,
        training=(not safety_pretraining))

    self._safety_critic_optimizer = safety_critic_optimizer
    self._lambda_optimizer = lambda_optimizer or alpha_optimizer
    if lambda_scheduler is None:
      self._lambda_scheduler = lambda_scheduler
    else:
      self._lambda_scheduler = lambda_scheduler(self._lambda_var)
    self._safety_pretraining = safety_pretraining
    self._safe_td_errors_loss_fn = safe_td_errors_loss_fn
    self._safety_gamma = safety_gamma or self._gamma
    self._fail_weight = fail_weight
    if train_critic_online:
      self._update_target_safety_critic = self._get_target_updater_safety_critic(
        tau=self._target_update_tau, period=self._target_update_period)
    else:
      self._update_target_safety_critic = None

  @property
  def safe_policy(self):
    return self._safe_policy

  def _get_target_updater_safety_critic(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target_safety_critic'):

      def update():
        """Update target network."""
        critic_update = common.soft_variables_update(
            self._safety_critic_network.variables,
            self._target_safety_critic_network.variables, tau)
        return critic_update

      return common.Periodically(update, period, 'update_target_safety_critic')

  def _initialize(self):
    """Returns an op to initialize the agent.

    Copies weights from the Q networks to the target Q network.
    """
    common.soft_variables_update(
        self._critic_network_1.variables,
        self._target_critic_network_1.variables,
        tau=1.0)
    common.soft_variables_update(
        self._critic_network_2.variables,
        self._target_critic_network_2.variables,
        tau=1.0)
    common.soft_variables_update(
      self._safety_critic_network.variables,
      self._target_safety_critic_network.variables,
      tau=1.0)

  def _experience_to_transitions(self, experience):
    boundary_mask = tf.logical_not(experience.is_boundary()[:, 0])
    experience = nest_utils.fast_map_structure(lambda *x: tf.boolean_mask(*x, boundary_mask), experience)
    squeeze_time_dim = not self._critic_network_1.state_spec
    time_steps, policy_steps, next_time_steps = (
        trajectory.experience_to_transitions(experience, squeeze_time_dim))
    return time_steps, policy_steps.action, next_time_steps  #, policy_steps.info

  def train_sc(self, experience, safe_rew, weights=None, metrics=None,
               training=True):
    """Returns a train op to update the agent's safety critic networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      safe_rew: Task-agnostic safety reward.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.
      metrics: Optional metrics to debug safety critic when computing loss
      training: Whether or not to apply gradient step to safety critic params

    Returns:
      A tuple of (safety_critic_loss, lambda_loss)

    """
    time_steps, actions, next_time_steps = (
        self._experience_to_transitions(experience))

    # update safety critic
    trainable_safety_variables = (
        self._safety_critic_network.trainable_variables)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_safety_variables, ('No trainable safety critic variables'
                                          ' to optimize.')
      tape.watch(trainable_safety_variables)
      safety_critic_loss = self.safety_critic_loss(
          time_steps,
          actions,
          next_time_steps,
          gamma=self._safety_gamma,
          safety_rewards=safe_rew,
          weights=weights,
          metrics=metrics)
    tf.debugging.check_numerics(safety_critic_loss, 'Critic loss is inf or '
                                'nan.')
    if training:
      safety_critic_grads = tape.gradient(safety_critic_loss,
                                          trainable_safety_variables)
      self._apply_gradients(safety_critic_grads, trainable_safety_variables,
                            self._safety_critic_optimizer)

    # update target safety critic independently of target critic during
    # pretraining if training critic online (i.e. on-policy data)
    if training and self._train_critic_online:
      self._update_target_safety_critic()

    lambda_variable = [self._lambda_var]
    next_actions, next_log_pis = self._actions_and_log_probs(
        time_steps, safety_constrained=True)

    # Dual gradient ascent step, updates lagrange multiplier
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert lambda_variable, 'No lambda to optimize'
      tape.watch(lambda_variable)
      lambda_loss = self.lambda_loss(
          time_steps,
          next_actions,
          safety_rewards=next_time_steps.observation['task_agn_rew'],
          weights=weights)
    tf.debugging.check_numerics(lambda_loss, 'Lambda loss is inf or nan.')

    if training:
      if self._lambda_scheduler is None:
          lambda_grads = tape.gradient(lambda_loss, lambda_variable)
          self._apply_gradients(lambda_grads, lambda_variable,
                                self._lambda_optimizer)
      else:
        self._lambda_scheduler()

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='lambda_loss', data=lambda_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='safety_critic_loss',
          data=safety_critic_loss,
          step=self.train_step_counter)

    return safety_critic_loss, lambda_loss

  def _train(self, experience, weights):
    """Returns a train op to update the agent's networks.

    This method trains with the provided batched experience.

    Args:
      experience: A time-stacked trajectory object.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      A train_op.

    Raises:
      ValueError: If optimizers are None and no default value was provided to
        the constructor.
    """
    time_steps, actions, next_time_steps = (
        self._experience_to_transitions(experience))

    trainable_critic_variables = (
        self._critic_network_1.trainable_variables +
        self._critic_network_2.trainable_variables)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss = self.critic_loss(
          time_steps,
          actions,
          next_time_steps,
          td_errors_loss_fn=self._td_errors_loss_fn,
          gamma=self._gamma,
          reward_scale_factor=self._reward_scale_factor,
          weights=weights)

    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self.actor_loss(time_steps, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    alpha_variable = [self._log_alpha]
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert alpha_variable, 'No alpha variable to optimize.'
      tape.watch(alpha_variable)
      alpha_loss = self.alpha_loss(time_steps, weights=weights)
    tf.debugging.check_numerics(alpha_loss, 'Alpha loss is inf or nan.')
    alpha_grads = tape.gradient(alpha_loss, alpha_variable)
    self._apply_gradients(alpha_grads, alpha_variable, self._alpha_optimizer)

    # updates safety critic if not training online
    safe_rew = next_time_steps.observation['task_agn_rew']
    sc_weight = None
    if self._fail_weight:
      sc_weight = tf.where(tf.cast(safe_rew, tf.bool), self._fail_weight / 0.5,
                           (1 - self._fail_weight) / 0.5)
    safety_critic_loss, lambda_loss = self.train_sc(
        experience, safe_rew, sc_weight,
        training=(not self._train_critic_online))

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
          name='critic_loss', data=critic_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='actor_loss', data=actor_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
          name='alpha_loss', data=alpha_loss, step=self.train_step_counter)
      if lambda_loss is not None:
        tf.compat.v2.summary.scalar(
            name='lambda_loss', data=lambda_loss, step=self.train_step_counter)
      if safety_critic_loss is not None:
        tf.compat.v2.summary.scalar(
            name='safety_critic_loss',
            data=safety_critic_loss,
            step=self.train_step_counter)

    self.train_step_counter.assign_add(1)
    self._update_target()

    total_loss = critic_loss + actor_loss + alpha_loss

    extra = SafeSacLossInfo(
        critic_loss=critic_loss, actor_loss=actor_loss, alpha_loss=alpha_loss,
        safety_critic_loss=safety_critic_loss, lambda_loss=lambda_loss)

    return tf_agent.LossInfo(loss=total_loss, extra=extra)

  def _apply_gradients(self, gradients, variables, optimizer):
    # list(...) is required for Python3.
    grads_and_vars = list(zip(gradients, variables))
    if self._gradient_clipping is not None:
      grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                       self._gradient_clipping)

    if self._summarize_grads_and_vars:
      eager_utils.add_variables_summaries(grads_and_vars,
                                          self.train_step_counter)
      eager_utils.add_gradients_summaries(grads_and_vars,
                                          self.train_step_counter)

    optimizer.apply_gradients(grads_and_vars)

  def _get_target_updater(self, tau=1.0, period=1):
    """Performs a soft update of the target network parameters.

    For each weight w_s in the original network, and its corresponding
    weight w_t in the target network, a soft update is:
    w_t = (1- tau) x w_t + tau x ws

    Args:
      tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
      period: Step interval at which the target network is updated.

    Returns:
      A callable that performs a soft update of the target network parameters.
    """
    with tf.name_scope('update_target'):

      def update():
        """Update target network."""
        critic_update_1 = common.soft_variables_update(
            self._critic_network_1.variables,
            self._target_critic_network_1.variables, tau)
        critic_update_2 = common.soft_variables_update(
            self._critic_network_2.variables,
            self._target_critic_network_2.variables, tau)
        if self._train_critic_online:
          return tf.group(critic_update_1, critic_update_2)
        critic_update_3 = common.soft_variables_update(
            self._safety_critic_network.variables,
            self._target_safety_critic_network.variables, tau)
        return tf.group(critic_update_1, critic_update_2, critic_update_3)

      return common.Periodically(update, period, 'update_targets')

  def _actions_and_log_probs(self, time_steps, safety_constrained=False):
    """Get actions and corresponding log probabilities from policy."""
    # Get raw action distribution from policy, and initialize bijectors list.
    batch_size = nest_utils.get_outer_shape(time_steps, self.time_step_spec)[0]
    policy = self.collect_policy
    policy_state = policy.get_initial_state(batch_size)
    action_distribution = policy.distribution(
      time_steps, policy_state=policy_state).action
    # Sample actions and log_pis from transformed distribution.
    if safety_constrained:
      actions, policy_state = self.safe_policy._apply_actor_network(
          time_steps.observation, time_steps.step_type, policy_state)
    else:
      actions = tf.nest.map_structure(lambda d: d.sample(), action_distribution)
    log_pi = common.log_probability(action_distribution, actions,
                                    self.action_spec)

    return actions, log_pi

  def safety_critic_loss(self,
                         time_steps,
                         actions,
                         next_time_steps,
                         safety_rewards,
                         gamma=1.,
                         weights=None,
                         metrics=None):
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
      tf.nest.assert_same_structure(actions, self.action_spec)
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)
      tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

      next_actions, next_log_pis = self._actions_and_log_probs(
        next_time_steps, safety_constrained=True)
      target_input = (next_time_steps.observation, next_actions)
      target_q_values, _ = self._target_safety_critic_network(
          target_input, next_time_steps.step_type)
      target_q_values = tf.nn.sigmoid(target_q_values)

      td_targets = tf.stop_gradient(safety_rewards +
                                    (1 - safety_rewards) * gamma *
                                    next_time_steps.discount * target_q_values)

      pred_input = (time_steps.observation, actions)
      pred_td_targets, _ = self._safety_critic_network(
          pred_input, time_steps.step_type, training=True)
      pred_td_targets = tf.nn.sigmoid(pred_td_targets)
      safety_critic_loss = self._safe_td_errors_loss_fn(td_targets,
                                                        pred_td_targets)

      if weights is not None:
        safety_critic_loss *= weights

      # Take the mean across the batch.
      safety_critic_loss = tf.reduce_mean(input_tensor=safety_critic_loss)

      if metrics:
        for metric in metrics:
          if isinstance(metric, tf.keras.metrics.AUC):
            metric.update_state(safety_rewards, pred_td_targets)
          else:
            rew_pred = tf.greater_equal(pred_td_targets,
                                        self._target_safety)
            metric.update_state(safety_rewards, rew_pred)

      if self._debug_summaries:
        td_errors = td_targets - pred_td_targets
        common.generate_tensor_summaries('safety_td_errors', td_errors,
                                         self.train_step_counter)
        common.generate_tensor_summaries('safety_td_targets', td_targets,
                                         self.train_step_counter)
        common.generate_tensor_summaries('safety_pred_td_targets',
                                         pred_td_targets,
                                         self.train_step_counter)

      return safety_critic_loss

  def critic_loss(self,
                  time_steps,
                  actions,
                  next_time_steps,
                  td_errors_loss_fn,
                  gamma=1.0,
                  reward_scale_factor=1.0,
                  weights=None):
    """Computes the critic loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      td_errors_loss_fn: A function(td_targets, predictions) to compute
        elementwise (per-batch-entry) loss.
      gamma: Discount for future rewards.
      reward_scale_factor: Multiplicative factor to scale rewards.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      tf.nest.assert_same_structure(actions, self.action_spec)
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)
      tf.nest.assert_same_structure(next_time_steps, self.time_step_spec)

      next_actions, next_log_pis = self._actions_and_log_probs(next_time_steps)
      target_input_1 = (next_time_steps.observation, next_actions)
      target_q_values1, _ = self._target_critic_network_1(
          target_input_1, next_time_steps.step_type, training=False)
      target_input_2 = (next_time_steps.observation, next_actions)
      target_q_values2, _ = self._target_critic_network_2(
          target_input_2, next_time_steps.step_type, training=False)
      target_q_values = (
          tf.minimum(target_q_values1, target_q_values2) -
          tf.exp(self._log_alpha) * next_log_pis)

      td_targets = tf.stop_gradient(
          reward_scale_factor * next_time_steps.reward +
          gamma * next_time_steps.discount * target_q_values)

      pred_input = (time_steps.observation, actions)
      pred_td_targets1, _ = self._critic_network_1(
          pred_input, time_steps.step_type, training=True)
      pred_td_targets2, _ = self._critic_network_2(
          pred_input, time_steps.step_type, training=True)
      critic_loss1 = td_errors_loss_fn(td_targets, pred_td_targets1)
      critic_loss2 = td_errors_loss_fn(td_targets, pred_td_targets2)
      critic_loss = critic_loss1 + critic_loss2

      if weights is not None:
        critic_loss *= weights

      # Take the mean across the batch.
      critic_loss = tf.reduce_mean(input_tensor=critic_loss)

      if self._debug_summaries:
        td_errors1 = td_targets - pred_td_targets1
        td_errors2 = td_targets - pred_td_targets2
        td_errors = tf.concat([td_errors1, td_errors2], axis=0)
        common.generate_tensor_summaries('td_errors', td_errors,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_targets', td_targets,
                                         self.train_step_counter)
        common.generate_tensor_summaries('pred_td_targets1', pred_td_targets1,
                                         self.train_step_counter)
        common.generate_tensor_summaries('pred_td_targets2', pred_td_targets2,
                                         self.train_step_counter)

      return critic_loss

  def actor_loss(self, time_steps, weights=None):
    """Computes the actor_loss for SAC training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)

      actions, log_pi = self._actions_and_log_probs(time_steps)
      target_input_1 = (time_steps.observation, actions)
      target_q_values1, _ = self._critic_network_1(target_input_1, time_steps.step_type, training=False)
      target_input_2 = (time_steps.observation, actions)
      target_q_values2, _ = self._critic_network_2(target_input_2, time_steps.step_type, training=False)
      target_q_values = tf.minimum(target_q_values1, target_q_values2)

      pred_input = (time_steps.observation, actions)
      q_val, _ = self._safety_critic_network(pred_input, time_steps.step_type)
      q_safe = tf.nn.sigmoid(q_val)
      # adds actor safety loss to actor_loss
      if self._safety_pretraining:
        actor_loss = tf.exp(self._log_alpha) * log_pi - target_q_values
      else:
        # safety_mask = tf.cast(q_safe > self._target_safety, tf.bool)
        # actor_loss = (
        #         tf.exp(self._log_alpha) * log_pi * q_safe - target_q_values +
        #         tf.exp(self._log_lambda) * (q_safe - self._target_safety))
        lam = tf.exp(self._lambda_var)
        actor_loss = (tf.exp(self._log_alpha) * log_pi - target_q_values +
                      tf.stop_gradient(lam * (q_safe - self._target_safety)))

      if weights is not None:
        actor_loss *= weights
      actor_loss = tf.reduce_mean(input_tensor=actor_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('actor_loss', actor_loss,
                                         self.train_step_counter)
        common.generate_tensor_summaries('actions', actions,
                                         self.train_step_counter)
        common.generate_tensor_summaries('log_pi', log_pi,
                                         self.train_step_counter)
        tf.compat.v2.summary.scalar(
          name='entropy_avg',
          data=-tf.reduce_mean(input_tensor=log_pi),
          step=self.train_step_counter)
        common.generate_tensor_summaries('target_q_values', target_q_values,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_safe', q_safe,
                                         self.train_step_counter)
        batch_size = nest_utils.get_outer_shape(time_steps,
                                                self._time_step_spec)[0]
        policy_state = self.policy.get_initial_state(batch_size)
        action_distribution = self.policy.distribution(time_steps,
                                                       policy_state).action
        if isinstance(action_distribution, tfp.distributions.Normal):
          common.generate_tensor_summaries('act_mean', action_distribution.loc,
                                           self.train_step_counter)
          common.generate_tensor_summaries('act_stddev',
                                           action_distribution.scale,
                                           self.train_step_counter)
        elif isinstance(action_distribution, tfp.distributions.Categorical):
          common.generate_tensor_summaries('act_mode',
                                           action_distribution.mode(),
                                           self.train_step_counter)
        try:
          common.generate_tensor_summaries('entropy_action',
                                           action_distribution.entropy(),
                                           self.train_step_counter)
        except NotImplementedError:
          pass  # Some distributions do not have an analytic entropy.

      return actor_loss

  def lambda_loss(self, time_steps, actions, safety_rewards, weights=None):
    """Computes the lambda_loss for SQRL training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      safety_rewards: Task-agnostic rewards for safety. 1 is unsafe, 0 is safe.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      lambda_loss: A scalar lambda_loss loss.
    """
    tf.nest.assert_same_structure(time_steps, self.time_step_spec)

    pred_input = (time_steps.observation, actions)
    q_val, unused_network_state1 = self._safety_critic_network(
        pred_input, time_steps.step_type)
    q_safe = tf.nn.sigmoid(q_val)

    lambda_loss = (
        self._lambda_var * tf.stop_gradient(-tf.math.log(q_safe) +
                                            tf.math.log(self._target_safety)))

    if weights is not None:
      lambda_loss *= weights

    lambda_loss = tf.reduce_mean(input_tensor=lambda_loss)

    with tf.name_scope('Losses'):
      tf.compat.v2.summary.scalar(
        name='lambda_loss', data=lambda_loss, step=self.train_step_counter)
      tf.compat.v2.summary.scalar(
        name='lambda', data=self._lambda_var, step=self.train_step_counter)

      return lambda_loss

  def alpha_loss(self, time_steps, weights=None):
    """Computes the alpha_loss for SQRL training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.

    Returns:
      alpha_loss: A scalar alpha loss.
    """
    with tf.name_scope('alpha_loss'):
      tf.nest.assert_same_structure(time_steps, self.time_step_spec)

      _, log_pi = self._actions_and_log_probs(time_steps)
      alpha_loss = (
          self._log_alpha * tf.stop_gradient(-log_pi - self._target_entropy))

      if weights is not None:
        alpha_loss *= weights

      alpha_loss = tf.reduce_mean(input_tensor=alpha_loss)

      if self._debug_summaries:
        common.generate_tensor_summaries('alpha_loss', alpha_loss,
                                         self.train_step_counter)

      return alpha_loss
