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

"""TF-Agents policies, networks, and helpers.

Custom TF-Agents policies, networks, and helpers for Safe SAC.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

import gin
import time
import numpy as np
import tensorflow as tf

from absl import logging
import tensorflow_probability as tfp
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import actor_policy, tf_policy
from tf_agents.policies import boltzmann_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.networks import normal_projection_network
from tf_agents.networks import utils
from tf_agents.distributions import utils as dist_utils

tfd = tfp.distributions

@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1,
                          state_dependent_std=True,
                          mean_transform=None,
                          scale_distribution=True):
  del init_action_stddev
  # std_bias_initializer_value = round(np.log(init_action_stddev + 1e-10), 3)
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      state_dependent_std=state_dependent_std,
      mean_transform=mean_transform,
      init_means_output_factor=init_means_output_factor,
      # std_bias_initializer_value=std_bias_initializer_value,
      # std_transform=std_clip_transform,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=scale_distribution)


@gin.configurable
def std_clip_transform(stddevs):
  stddevs = tf.nest.map_structure(lambda t: tf.clip_by_value(t, -20, 2),
                                  stddevs)
  return tf.exp(stddevs)


@gin.configurable
class CriticEncoderNetwork(network.Network):
  """Critic Network with encoding networks for observation and action."""

  def __init__(
      self,
      input_tensor_spec,
      observation_preprocessing_combiner=None,
      observation_conv_layer_params=None,
      observation_fc_layer_params=None,
      observation_dropout_layer_params=None,
      action_fc_layer_params=None,
      action_dropout_layer_params=None,
      joint_preprocessing_combiner=None,
      joint_fc_layer_params=None,
      joint_dropout_layer_params=None,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform'),
      activation_fn=tf.nn.relu,
      name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      joint_preprocessing_combiner: Combiner layer for obs and action inputs
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    observation_spec, action_spec = input_tensor_spec

    if (len(tf.nest.flatten(observation_spec)) > 1 and
        joint_preprocessing_combiner is None and observation_preprocessing_combiner is None):
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    preprocessing_layers = None
    # combiner assumes a single batch dimension, without time

    super(CriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    if (observation_preprocessing_combiner or observation_conv_layer_params or
        observation_fc_layer_params or observation_dropout_layer_params):
      self._obs_encoder = encoding_network.EncodingNetwork(
        observation_spec,
        preprocessing_combiner=observation_preprocessing_combiner,
        conv_layer_params=observation_conv_layer_params,
        fc_layer_params=observation_fc_layer_params,
        dropout_layer_params=observation_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
      observation_spec = tensor_spec.TensorSpec(self._obs_encoder._postprocessing_layers.output_shape,
                                                name='obs_enc')
    else:
      self._obs_encoder = None

    if (action_fc_layer_params or action_dropout_layer_params):
      self._ac_encoder = encoding_network.EncodingNetwork(
        action_spec,
        fc_layer_params=action_fc_layer_params,
        dropout_layer_params=action_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
      action_spec = tensor_spec.TensorSpec(self._ac_encoder._postprocessing_layers.output_shape,
                                           name='ac_enc')
    else:
      self._ac_encoder = None

    input_tensor_spec = (observation_spec, action_spec)
    self._encoder = encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=joint_preprocessing_combiner,
        fc_layer_params=joint_fc_layer_params,
        dropout_layer_params=joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
    self._value_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003),
        name='value')


@gin.configurable
class CriticNetwork(network.Network):
  """CriticNetwork implemented with encoder network"""

  def __init__(
      self,
      input_tensor_spec,
      preprocessing_combiner=None,
      joint_fc_layer_params=None,
      joint_dropout_layer_params=None,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform'),
      activation_fn=tf.nn.relu,
      name='CriticNetwork'):
    """Creates an instance of `CriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      preprocessing_combiner: Combiner layer for obs and action inputs
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    observation_spec, action_spec = input_tensor_spec

    if (len(tf.nest.flatten(observation_spec)) > 1 and
        preprocessing_combiner is None):
      raise ValueError('Only a single observation is supported by this network')

    flat_action_spec = tf.nest.flatten(action_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]

    preprocessing_layers = None
    # combiner assumes a single batch dimension, without time

    super(CriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, state_spec=(), name=name)

    self._encoder = encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=joint_fc_layer_params,
        dropout_layer_params=joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
    self._value_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003),
        bias_initializer=tf.keras.initializers.RandomUniform(-1, 0),
        name='value')

  def call(self, observations, step_type, network_state=(), training=False):
    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state,
        training=training)
    q_val = self._value_layer(state)
    return tf.reshape(q_val, [-1]), network_state


#### WCPG classes


def _critic_normal_projection_net(output_spec,
                                  init_stddev=0.35,
                                  init_means_output_factor=0.01):
  std_bias_initializer_value = round(np.log(init_action_stddev + 1e-10), 3)

  return normal_projection_network.NormalProjectionNetwork(
      output_spec,
      init_means_output_factor=init_means_output_factor,
      std_bias_initializer_value=std_bias_initializer_value,
      mean_transform=None,
      std_transform=sac_agent.std_clip_transform,
      state_dependent_std=True,
      scale_distribution=False)


@gin.configurable
class DistributionalCriticNetwork(network.DistributionNetwork):
  """DistributionalCriticNetwork implemented with encoder networks"""

  def __init__(
      self,
      input_tensor_spec,
      preprocessing_layer_size=64,
      joint_fc_layer_params=(64,),
      joint_dropout_layer_params=None,
      kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
          scale=1. / 3., mode='fan_in', distribution='uniform'),
      activation_fn=tf.nn.relu,
      name='DistributionalCriticNetwork'):
    """Creates an instance of `DistributionalCriticNetwork`.

    Args:
      input_tensor_spec: A tuple of (observation, action) each a nest of
        `tensor_spec.TensorSpec` representing the inputs.
      joint_fc_layer_params: Optional list of fully connected parameters after
        merging observations and actions, where each item is the number of units
        in the layer.
      joint_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent', if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of joint_fc_layer_params, or
        be None.
      kernel_initializer: Initializer to use for the kernels of the conv and
        dense layers. If none is provided a default glorot_uniform
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      name: A string representing name of the network.

    Raises:
      ValueError: If `observation_spec` or `action_spec` contains more than one
        observation.
    """
    assert len(input_tensor_spec) == 3, 'input_tensor_spec should contain obs, ac, and alpha specs'
    observation_spec, action_spec, alpha_spec = input_tensor_spec

    preprocessing_layers = (tf.keras.layers.Dense(preprocessing_layer_size),
                            tf.keras.layers.Lambda(lambda x: x),
                            tf.keras.layers.Dense(preprocessing_layer_size))
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    output_spec = tensor_spec.TensorSpec(shape=(1,), name='R')

    super(DistributionalCriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, output_spec=output_spec, state_spec=(), name=name)

    self._encoder = encoding_network.EncodingNetwork(
        input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=joint_fc_layer_params,
        dropout_layer_params=joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
    self._projection_network = _critic_normal_projection_net(output_spec)

  def call(self, observations, step_type, network_state=(), training=False):
    state, network_state = self._encoder(
      observations,
      step_type=step_type,
      network_state=network_state,
      training=training)
    outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
    output_actions = self._projection_network(state, outer_rank)
    return output_actions, network_state


@gin.configurable
class WcpgActorNetwork(network.Network):
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layer_size=32,
               fc_layer_params=(32,),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                  scale=1. / 3., mode='fan_in', distribution='uniform'),
               batch_squash=True,
               name='WcpgActorNetwork'):
    super(WcpgActorNetwork, self).__init__(
      input_tensor_spec=input_tensor_spec,
      state_spec=(),
      name=name)
    preprocessing_layers = [tf.keras.layers.Dense(preprocessing_layer_size),
                            tf.keras.layers.Dense(preprocessing_layer_size/2)]
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)
    observation_spec, alpha_spec = input_tensor_spec

    self._output_tensor_spec = output_tensor_spec
    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]
    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    self._alpha_spec = alpha_spec
    self._encoder = encoding_network.EncodingNetwork(
      input_tensor_spec,
      preprocessing_layers=preprocessing_layers,
      preprocessing_combiner=preprocessing_combiner,
      fc_layer_params=fc_layer_params,
      dropout_layer_params=dropout_layer_params,
      activation_fn=activation_fn,
      kernel_initializer=kernel_initializer,
      batch_squash=batch_squash)
    self._action_layer = tf.keras.layers.Dense(flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.003, maxval=0.003),
            name='action')

  def call(self, observations, step_type=(), network_state=(), training=False):
    state, network_state = self._encoder(observations, step_type=step_type,
                                         network_state=network_state, training=training)
    return self._action_layer(state), network_state

  @property
  def alpha_spec(self):
    return self._alpha_spec


WcpgPolicyInfo = collections.namedtuple('WcpgPolicyInfo', ('alpha',))


@gin.configurable
class WcpgPolicy(actor_policy.ActorPolicy):
  """A policy that awares safety."""

  def __init__(self, time_step_spec, action_spec, actor_network, alpha=None,
               alpha_sampler=None, observation_normalizer=None, clip=True,
               training=False, name="WcpgPolicy"):
    info_spec = WcpgPolicyInfo(alpha=actor_network.alpha_spec)
    super(WcpgPolicy, self).__init__(time_step_spec, action_spec, actor_network, info_spec,
                                     observation_normalizer, clip, training, name)
    self._alpha = alpha
    self._alpha_sampler = alpha_sampler or (lambda: np.random.uniform(0.1, 1.))

  @property
  def alpha(self):
    if self._alpha is None:
      self._alpha = self._alpha_sampler()
    return self._alpha

  def _apply_actor_network(self, time_step, policy_state):
    observation = time_step.observation

    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    if tf.is_tensor(observation):
      if not nest_utils.is_batched_nested_tensors(observation, self.time_step_spec.observation):
        observation = nest_utils.batch_nested_tensors(observation)
    else:
      if not nest_utils.get_outer_array_shape(observation, self.time_step_spec.observation):
        observation = nest_utils.batch_nested_array(observation)

    alpha = np.array([self.alpha])[None]
    return self._actor_network((observation, alpha), time_step.step_type, policy_state,
                               training=self._training)

  def _distribution(self, time_step, policy_state):
    if time_step.is_first():
      self._alpha = None
    distribution_step = super(WcpgPolicy, self)._distribution(time_step, policy_state)
    return distribution_step._replace(info=WcpgPolicyInfo(alpha=self.alpha))


class WcpgPolicyWrapper(tf_policy.Base):
  def __init__(self, wrapped_policy, alpha=None, alpha_sampler=None, clip=True, name=None):
    alpha_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, minimum=0., maximum=1.,
                                               name='alpha')
    super(WcpgPolicyWrapper, self).__init__(
      wrapped_policy.time_step_spec,
      wrapped_policy.action_spec,
      wrapped_policy.policy_state_spec,
      WcpgPolicyInfo(alpha=alpha_spec),
      clip=clip,
      name=name)
    self._wrapped_policy = wrapped_policy
    self._alpha = alpha
    self._alpha_sampler = alpha_sampler or (lambda: np.random.uniform(0.1, 1.))

  @property
  def alpha(self):
    if self._alpha is None:
      self._alpha = self._alpha_sampler()
    return self._alpha

  def _action(self, time_step, policy_state, seed):
    if time_step.is_first():
      self._alpha = None
    policy_step = self._wrapped_policy.action(time_step, policy_state, seed)
    return policy_step._replace(info=WcpgPolicyInfo(alpha=self.alpha))


class GaussianNoisePolicy(tf_policy.Base):
  def __init__(self,
               wrapped_policy,
               exploration_noise_stddev=1.0,
               clip=True,
               name=None):

    def _validate_action_spec(action_spec):
      if not tensor_spec.is_continuous(action_spec):
        raise ValueError('Gaussian Noise is applicable only to continuous actions.')

    tf.nest.map_structure(_validate_action_spec, wrapped_policy.action_spec)
    super(GaussianNoisePolicy, self).__init__(
        wrapped_policy.time_step_spec,
        wrapped_policy.action_spec,
        wrapped_policy.policy_state_spec,
        wrapped_policy.info_spec,
        clip=clip,
        name=name)
    self._exploration_noise_stddev = exploration_noise_stddev
    self._wrapped_policy = wrapped_policy

  def _variables(self):
    return self._wrapped_policy.variables()

  def _action(self, time_step, policy_state, seed):
    seed_stream = tfd.SeedStream(seed=seed, salt='gaussian_noise')
    action_step = self._wrapped_policy.action(time_step, policy_state, seed_stream())
    actions = action_step.action
    actions += tf.random.normal(stddev=self._exploration_noise_stddev,
                                shape=actions.shape,
                                dtype=actions.dtype)
    return policy_step.PolicyStep(actions, action_step.state, action_step.info)


@gin.configurable
class SafeActorPolicyRSVar(actor_policy.ActorPolicy):
  """Returns safe actions by rejection sampling with increasing variance."""

  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               safety_critic_network=None,
               safety_threshold=0.1,
               info_spec=(),
               observation_normalizer=None,
               clip=True,
               resample_counter=None,
               resample_n=50,
               resample_k=5,
               training=False,
               name=None):
    super(SafeActorPolicyRSVar,
          self).__init__(time_step_spec, action_spec, actor_network, info_spec,
                         observation_normalizer, clip, training, name)
    self._safety_critic_network = safety_critic_network
    self._safety_threshold = safety_threshold
    self._resample_counter = resample_counter
    self._n = resample_n
    self._k = resample_k

  def _apply_actor_network(self, time_step, policy_state):
    has_batch_dim = time_step.step_type.shape.as_list()[0] is None or time_step.step_type.shape.as_list()[0] > 1
    observation = time_step.observation
    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    actions, policy_state = self._actor_network(observation,
                                                time_step.step_type,
                                                policy_state,
                                                training=self._training)
    if has_batch_dim or self._training:  # returns normal actions, unmasked, when not training
      return actions, policy_state

    # samples "best" safe action out of 50
    n, k = self._n, self._k
    sampled_ac = actions.sample(n)
    obs = nest_utils.stack_nested_tensors(
        [time_step.observation for _ in range(n)])

    obs_outer_rank = nest_utils.get_outer_rank(obs, self.time_step_spec.observation)
    ac_outer_rank = nest_utils.get_outer_rank(sampled_ac, self.action_spec)
    obs_batch_squash = utils.BatchSquash(obs_outer_rank)
    ac_batch_squash = utils.BatchSquash(ac_outer_rank)
    obs = tf.nest.map_structure(obs_batch_squash.flatten, obs)
    sampled_ac = tf.nest.map_structure(ac_batch_squash.flatten, sampled_ac)

    q_val, _ = self._safety_critic_network((obs, sampled_ac),
                                           time_step.step_type)
    fail_prob = tf.nn.sigmoid(q_val)
    safe_ac_idx = tf.where(fail_prob < self._safety_threshold)

    resample_count = 0
    while self._training and resample_count < k and not safe_ac_idx.shape.as_list()[0]:
      if self._resample_counter is not None:
        self._resample_counter()
      resample_count += 1
      if isinstance(actions, dist_utils.SquashToSpecNormal):
        scale = actions.input_distribution.scale * 1.5  # increase variance by constant 1.5
        ac_mean = actions.mean()
      else:
        scale = actions.scale * 1.5
        ac_mean = actions.mean()
      actions = self._actor_network.output_spec.build_distribution(
          loc=ac_mean, scale=scale)
      sampled_ac = actions.sample(n)
      sampled_ac = tf.nest.map_structure(ac_batch_squash.flatten, sampled_ac)
      q_val, _ = self._safety_critic_network((obs, sampled_ac),
                                             time_step.step_type)

      fail_prob = tf.nn.sigmoid(q_val)
      safe_ac_idx = tf.where(fail_prob < self._safety_threshold)
    # logging.debug('resampled {} times, {} seconds'.format(resample_count, time.time() - start_time))
    sampled_ac = ac_batch_squash.unflatten(sampled_ac)
    if None in safe_ac_idx.shape.as_list() or not np.prod(safe_ac_idx.shape.as_list()):
      safe_idx = tf.argmin(fail_prob)  # return action closest to fail prob eps
    else:
      sampled_ac = tf.gather(sampled_ac, safe_ac_idx)
      fail_prob_safe = tf.gather(fail_prob, tf.squeeze(safe_ac_idx))
      if self._training:
        safe_idx = tf.argmax(fail_prob_safe)  # picks most unsafe action out of "safe" options
      else:
        # picks random safe_action
        log_prob = common.log_probability(actions, sampled_ac, self.action_spec)
        safe_idx = tfp.distributions.Categorical(log_prob).sample()
        safe_idx = tf.reshape(safe_idx, [-1])[0]
        # safe_idx = tf.argmin(fail_prob_safe)[0]  # picks the safest action
    ac = sampled_ac[safe_idx]
    assert ac.shape.as_list()[0] == 1, 'action shape is not correct: {}'.format(ac.shape.as_list())
    return ac, policy_state


BoltzmannPolicyInfo = collections.namedtuple('BoltzmannPolicyInfo',
                                             ('temperature',))

@gin.configurable
class SafetyBoltzmannPolicy(boltzmann_policy.BoltzmannPolicy):
  """A policy that awares safety."""

  def __init__(self, policy, temperature=1.0, name=None):
    super(SafetyBoltzmannPolicy, self).__init__(policy, temperature, name)
    info_spec = BoltzmannPolicyInfo(
        temperature=tensor_spec.TensorSpec((), tf.float32, name='temperature'))
    self._info_spec = info_spec
    self._setup_specs()  # run again to make sure specs are correctly updated

  def _distribution(self, time_step, policy_state):
    distribution_step = super(SafetyBoltzmannPolicy,
                              self)._distribution(time_step, policy_state)
    distribution_step = distribution_step._replace(
        info=BoltzmannPolicyInfo(temperature=self._temperature))
    return distribution_step
