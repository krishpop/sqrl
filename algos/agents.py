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
import pdb

from absl import logging
import tensorflow_probability as tfp
from safemrl.utils import misc
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import encoding_network
from tf_agents.networks import network
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
                          scale_distribution=True):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
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
      value_bias_initializer=None,
      activation_fn=tf.nn.relu,
      value_activation_fn=None,
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
        activation=value_activation_fn,
        kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.003, maxval=0.003),
        bias_initializer=value_bias_initializer,
        name='value')

  def call(self, observations, step_type, network_state=(), training=False, mask=None):
    state, network_state = self._encoder(
        observations, step_type=step_type, network_state=network_state,
        training=training)
    q_val = self._value_layer(state)
    return tf.reshape(q_val, [-1]), network_state


#### WCPG classes


def _critic_normal_projection_net(output_spec,
                                  init_stddev=0.35,
                                  init_means_output_factor=0.1):
  del init_stddev
  # std_bias_initializer_value = round(np.log(init_action_stddev + 1e-10), 3)

  return normal_projection_network.NormalProjectionNetwork(
      output_spec,
      init_means_output_factor=init_means_output_factor,
      # std_bias_initializer_value=std_bias_initializer_value,
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
      obs_preprocessing_combiner=misc.extract_observation_layer,
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
        None. This list must have the same length of joint_fc_layeri_params, or
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

    preprocessing_combiner = misc.concatenate_lambda_layer()
 
    output_spec = tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32, name='R')

    super(DistributionalCriticNetwork, self).__init__(
        input_tensor_spec=input_tensor_spec, output_spec=output_spec, state_spec=(), name=name)

    pre_obs = tensor_spec.TensorSpec(shape=(preprocessing_layer_size,), dtype=tf.float32, name='pre_o')
    pre_alph = tensor_spec.TensorSpec(shape=(preprocessing_layer_size//2,), dtype=tf.float32,
                                      name='pre_alph')

    self._obs_encoder = encoding_network.EncodingNetwork(
        observation_spec, preprocessing_combiner=obs_preprocessing_combiner(),
        fc_layer_params=(preprocessing_layer_size,), kernel_initializer=kernel_initializer
    )
    self._alph_encoder = encoding_network.EncodingNetwork(
      alpha_spec, fc_layer_params=(preprocessing_layer_size//2,), kernel_initializer=kernel_initializer
    )

    self.encoder_input_tensor_spec = (pre_obs, action_spec, pre_alph)

    self._encoder = encoding_network.EncodingNetwork(
        self.encoder_input_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=joint_fc_layer_params,
        dropout_layer_params=joint_dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)
    self._projection_network = _critic_normal_projection_net(output_spec)

  def call(self, observations, step_type, network_state=(), training=False):
    obs, ac, alpha = observations
    pre_obs, _ = self._obs_encoder(obs, step_type=step_type, network_state=network_state,
                                   training=training)
    pre_alpha, _ = self._alph_encoder(alpha, step_type=step_type, network_state=network_state,
                                       training=training)
    observations = (pre_obs, ac, pre_alpha)
    state, network_state = self._encoder(
      observations,
      step_type=step_type,
      network_state=network_state,
      training=training)
    outer_rank = nest_utils.get_outer_rank(observations, self.encoder_input_tensor_spec)
    q_distribution, _ = self._projection_network(state, outer_rank)
    return q_distribution, network_state


@gin.configurable
class WcpgActorNetwork(network.Network):
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               obs_preprocessing_combiner=misc.extract_observation_layer,
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

    observation_spec, alpha_spec = input_tensor_spec

    pre_obs = tensor_spec.TensorSpec(shape=(preprocessing_layer_size,), dtype=tf.float32, name='pre_o')
    pre_alph = tensor_spec.TensorSpec(shape=(preprocessing_layer_size//2,), dtype=tf.float32,
                                      name='pre_alph')

    self._obs_encoder = encoding_network.EncodingNetwork(
        observation_spec, preprocessing_combiner=obs_preprocessing_combiner(),
        fc_layer_params=(preprocessing_layer_size,), kernel_initializer=kernel_initializer
    )
    self._alph_encoder = encoding_network.EncodingNetwork(
      alpha_spec, fc_layer_params=(preprocessing_layer_size//2,), kernel_initializer=kernel_initializer
    )

    self.encoder_input_tensor_spec = (pre_obs, pre_alph)

    self._output_tensor_spec = output_tensor_spec
    flat_action_spec = tf.nest.flatten(output_tensor_spec)
    if len(flat_action_spec) > 1:
      raise ValueError('Only a single action is supported by this network')
    self._single_action_spec = flat_action_spec[0]
    if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
      raise ValueError('Only float actions are supported by this network.')

    self._alpha_spec = alpha_spec
    preprocessing_combiner = misc.concatenate_lambda_layer()

    self._encoder = encoding_network.EncodingNetwork(
      self.encoder_input_tensor_spec,
      preprocessing_layers=None,
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
    obs, alph = observations
    pre_obs, _ = self._obs_encoder(obs, step_type=step_type,
                                   network_state=network_state, training=training)
    pre_alph, _ = self._alph_encoder(alph, step_type=step_type,
                                     network_state=network_state, training=training)
    observations = (pre_obs, pre_alph)
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
    super(WcpgPolicy, self).__init__(time_step_spec=time_step_spec, action_spec=action_spec,
                                     actor_network=actor_network, info_spec=info_spec,
                                     observation_normalizer=observation_normalizer,
                                     clip=clip, training=training, name=name)
    self._alpha = alpha
    self._alpha_sampler = alpha_sampler or (lambda: np.random.uniform(0.1, 1.))

  @property
  def alpha(self):
    if self._alpha is None:
      self._alpha = self._alpha_sampler()
    return self._alpha

  def _apply_actor_network(self, time_step, step_type, policy_state, mask=None):
    observation = time_step

    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    if tf.is_tensor(observation):
      if not nest_utils.is_batched_nested_tensors(observation, self.time_step_spec.observation):
        observation = nest_utils.batch_nested_tensors(observation)
    else:
      if not nest_utils.get_outer_array_shape(observation, self.time_step_spec.observation):
        observation = nest_utils.batch_nested_array(observation)

    alpha = np.array([self.alpha])[None]
    return self._actor_network((observation, alpha), step_type, policy_state,
                               training=self._training)

  def _distribution(self, time_step, policy_state):
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


def resample_cond(scale, safe_ac_mask, *_):
  return tf.logical_not(tf.reduce_any(tf.cast(safe_ac_mask, tf.bool)))


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
               resample_k=6,
               training=False,
               sampling_method='rejection',
               name=None):
    super(SafeActorPolicyRSVar,
          self).__init__(time_step_spec, action_spec, actor_network, info_spec,
                         observation_normalizer, clip, training, name)
    self._safety_critic_network = safety_critic_network
    self._safety_threshold = safety_threshold
    self._resample_counter = resample_counter
    self._n = resample_n
    self._k = resample_k
    self._sample_action = common.function_in_tf1()(self._resample_action_fn)
    self._sampling_method = sampling_method

  def _loop_body_fn(self, ac_batch_squash, obs, step_type, ac_mean):
    def loop_body(scale, safe_ac_mask, sampled_ac, fail_prob):
      if self._resample_counter is not None:
        self._resample_counter()

      actions = self._actor_network.output_spec.build_distribution(loc=ac_mean, scale=scale)
      sampled_ac = actions.sample(self._n)
      sampled_ac = tf.nest.map_structure(ac_batch_squash.flatten, sampled_ac)
      q_val, _ = self._safety_critic_network((obs, sampled_ac), step_type)

      fail_prob = tf.nn.sigmoid(q_val)
      safe_ac_mask = fail_prob < self._safety_threshold
      return [scale * 1.5, safe_ac_mask, sampled_ac, fail_prob]
    return loop_body

  def _resample_action_fn(self, resample_input):
    ac_mean, scale, step_type, *flat_observation = resample_input  # expects single ac, obs, step
    n, k = self._n, self._k

    # samples "best" safe action out of 50
    # sampled_ac = actions.sample(n)
    observation = tf.nest.pack_sequence_as(self.time_step_spec.observation, flat_observation)
    obs = nest_utils.stack_nested_tensors([observation for _ in range(n)])

    actions = self._actor_network.output_spec.build_distribution(loc=ac_mean,
                                                                 scale=scale)
    sampled_ac = actions.sample(n)

    ac_outer_rank = nest_utils.get_outer_rank(sampled_ac, self.action_spec)
    ac_batch_squash = utils.BatchSquash(ac_outer_rank)
    sampled_ac = tf.nest.map_structure(ac_batch_squash.flatten, sampled_ac)

    obs_outer_rank = nest_utils.get_outer_rank(obs, self.time_step_spec.observation)
    obs_batch_squash = utils.BatchSquash(obs_outer_rank)
    obs = tf.nest.map_structure(obs_batch_squash.flatten, obs)

    q_val, _ = self._safety_critic_network((obs, sampled_ac), step_type)
    fail_prob = tf.nn.sigmoid(q_val)
    safe_ac_mask = fail_prob < self._safety_threshold

    # pdb.set_trace()
    [_, safe_ac_mask, sampled_ac, fail_prob] = tf.while_loop(
      cond=resample_cond,
      body=self._loop_body_fn(ac_batch_squash, obs, step_type, ac_mean),
      loop_vars=[scale, safe_ac_mask, sampled_ac, fail_prob],
      maximum_iterations=k
    )
    sampled_ac = tf.nest.map_structure(ac_batch_squash.unflatten, sampled_ac)

    if self._resample_counter is not None:
      logging.debug('resampled %d times', self._resample_counter.result())

    safe_ac_idx = tf.where(safe_ac_mask)
    fail_prob_safe = tf.gather(fail_prob, safe_ac_idx[:, 0])
    safe_idx = self._get_safe_idx(safe_ac_mask, fail_prob, sampled_ac,
                                  safe_ac_idx, actions, fail_prob_safe)
    ac = sampled_ac[safe_idx]
    return ac

  @common.function(autograph=True)
  def _get_safe_idx(self, safe_ac_mask, fail_prob, sampled_ac, safe_ac_idx,
                   actions, fail_prob_safe):
    if tf.math.count_nonzero(safe_ac_mask) == 0:
      # picks safest action
      safe_idx = tf.argmin(fail_prob)
    else:
      sampled_ac = tf.gather(sampled_ac, safe_ac_idx)
      # picks most unsafe "safe" action
      # safe_idx = tf.argmax(fail_prob_safe, axis=0)

      # picks the safest action
      # safe_idx = tf.argmin(fail_prob_safe)

      if self._training:
        # picks random safe_action, weighted by 1 - fail_prob_safe (so higher weight for safer actions)
        # safe_idx = tfp.distributions.Categorical([1 - fail_prob_safe]).sample()
        if self._sampling_method == 'rejection':
          # standard rejection sampling with prob proportional to original policy
          log_prob = common.log_probability(actions, sampled_ac, self.action_spec)
          safe_idx = tfp.distributions.Categorical(log_prob).sample()
        elif self._sampling_method == 'risky':
          # picks random risky safe action, weighted by fail_prob_safe (so higher weight for less safe actions)
          safe_idx = tfp.distributions.Categorical([fail_prob_safe]).sample()
        elif self._sampling_method == 'safe':
          safe_idx = tfp.distributions.Categorical([1-fail_prob_safe]).sample()
      safe_idx = tf.reshape(safe_idx, [-1])[0]
    return safe_idx

  def _apply_actor_network(self, observation, step_type, policy_state, mask=None):
    if observation['observation'].shape.as_list()[0] is None:
      has_batch_dim = True
    else:
      has_batch_dim = observation['observation'].shape.as_list()[0] > 1

    if self._observation_normalizer:
      observation = self._observation_normalizer.normalize(observation)
    actions, policy_state = self._actor_network(observation,
                                                step_type,
                                                policy_state,
                                                training=self._training)
    # EDIT 5/18 - training now determines whether safe/unsafe actions are sampled
    # returns normal actions, unmasked, when not training
    if not self._training:
      return actions, policy_state

    # setup input for sample_action
    ac_mean = actions.mean()
    if isinstance(actions, dist_utils.SquashToSpecNormal):
      scale = actions.input_distribution.scale
    else:
      scale = actions.scale

    if has_batch_dim:
      ac = tf.map_fn(self._sample_action, [ac_mean, scale, step_type] + list(tf.nest.flatten(observation)),
                     dtype=tf.float32)
    else:
      ac = self._sample_action([ac_mean, scale, step_type] + list(tf.nest.flatten(observation)))
    if ac is None:
      return actions, policy_state

    if has_batch_dim:
      ac = tf.squeeze(ac)
    if not has_batch_dim:
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
