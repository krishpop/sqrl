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
"""Custom TFAgent PyMetric for minitaur and point-mass environments.

AverageEarlyFailureMetric used for detecting fall count for minitaur env, and
AverageFallenMetric and AverageSuccessMetric used for poit-mass envs.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import numpy as np

from tf_agents.metrics import py_metrics
from tf_agents.utils import numpy_storage


@gin.configurable
class AverageEarlyFailureMetric(py_metrics.StreamingMetric):
  """Computes average early failure rate in buffer_size episodes."""

  def __init__(self,
               max_episode_len=500,
               name='AverageEarlyFailure',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageEnvObsDict."""
    self._np_state = numpy_storage.NumpyState()
    self._max_episode_len = max_episode_len
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_steps = np.array(0, dtype=np.int32)
    super(AverageEarlyFailureMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_steps = np.zeros(shape=(batch_size,), dtype=np.int32)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_steps = self._np_state.episode_steps
    is_last = np.where(trajectory.is_boundary())
    not_last = np.where(~trajectory.is_boundary())

    episode_steps[not_last] += 1
    if len(is_last[0]) > 0:
      self.add_to_buffer(episode_steps[is_last] < self._max_episode_len)
    episode_steps[is_last] = 0


# TODO: add max episode length to these metrics
@gin.configurable
class AverageFallenMetric(py_metrics.StreamingMetric):
  """Computes average fallen rate for PointMass envs in buffer_size episodes."""

  def __init__(self,
               dtype=np.bool,
               name='AverageFallen',
               buffer_size=10,
               batch_size=None):
    """Creates an AverageFallenMetric."""
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._dtype = dtype
    super(AverageFallenMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_boundary())

    if len(is_last[0]) > 0:
      self.add_to_buffer(trajectory.observation['task_agn_rew'][is_last])


@gin.configurable
class TotalFallenMetric(py_metrics.CounterMetric):
  def __init__(self, name='TotalFallen'):
    super(TotalFallenMetric, self).__init__(name)

  def call(self, trajectory):
    if trajectory.is_boundary():
      self._np_state.count += 1. * trajectory.observation['task_agn_rew'][0]


@gin.configurable
class TotalSuccessMetric(py_metrics.CounterMetric):
  def __init__(self, name='TotalSuccess'):
    super(TotalSuccessMetric, self).__init__(name)

  def call(self, trajectory):
    if trajectory.is_last():
      self._np_state.count += 1 * (trajectory.reward >= 1.)[0]


@gin.configurable
class AverageSuccessMetric(py_metrics.StreamingMetric):
  """Computes average success rate for PointMass env in buffer_size episodes."""

  def __init__(self, name='AverageSuccess', buffer_size=10, batch_size=None):
    """Creates an AverageSuccessMetric."""
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    super(AverageSuccessMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_last())

    if len(is_last[0]) > 0:
      succ = np.logical_and(
          np.logical_not(trajectory.observation['task_agn_rew'][is_last]),
          trajectory.reward[is_last] >= 1.)
      self.add_to_buffer(succ)


@gin.configurable
class MinitaurAverageSpeedMetric(py_metrics.StreamingMetric):
  """Computes average early failure rate in buffer_size episodes."""

  def __init__(self,
               name='MinitaurAverageSpeed',
               buffer_size=10,
               batch_size=None):
    """Creates a metric for minitaur speed stats."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.episode_steps = np.array(0, dtype=float)
    self._np_state.speed = np.array(0, dtype=float)
    super(MinitaurAverageSpeedMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.episode_steps = np.zeros(shape=(batch_size,), dtype=np.int32)
    self._np_state.speed = np.zeros(shape=(batch_size,), dtype=float)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    episode_steps = self._np_state.episode_steps
    total_speed = self._np_state.speed
    is_last = np.where(trajectory.is_boundary())
    not_last = np.where(~trajectory.is_boundary())
    total_speed[not_last] += trajectory.observation['current_vel'][~trajectory.is_boundary()]
    episode_steps[not_last] += 1

    if len(is_last[0]) > 0:
      self.add_to_buffer(total_speed[is_last]/episode_steps[is_last])
    episode_steps[is_last] = 0
    total_speed[is_last] = 0


@gin.configurable
class MinitaurAverageMaxSpeedMetric(py_metrics.StreamingMetric):
  """Computes average early failure rate in buffer_size episodes."""

  def __init__(self,
               name='MinitaurAverageMaxSpeed',
               buffer_size=10,
               batch_size=None):
    """Creates a metric for minitaur speed stats."""
    self._np_state = numpy_storage.NumpyState()
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._np_state.speed = np.array(0, dtype=float)
    super(MinitaurAverageMaxSpeedMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    """Resets stat gathering variables."""
    self._np_state.speed = np.zeros(shape=(batch_size,), dtype=float)

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """
    max_speed = self._np_state.speed
    is_last = np.where(trajectory.is_boundary())
    not_last = np.where(~trajectory.is_boundary())
    if len(not_last[0]) > 0:
      max_speed[not_last] = np.max([trajectory.observation['current_vel'][not_last],
                                    max_speed[not_last]], axis=0)

    if len(is_last[0]) > 0:
      self.add_to_buffer(max_speed[is_last])
    max_speed[is_last] = 0


@gin.configurable
class CubeAverageScoreMetric(py_metrics.StreamingMetric):
  """Computes average score at end of trajectory"""
  def __init__(self, env, name='AverageScore', buffer_size=10, batch_size=None):
    """
    Creates an CubeAverageScoreMetric.
    Args:
      env: Instance of gym.Env that implements get_score() which updates the metric
      name: metric name
      buffer_size: number of episodes to compute average over
    """

    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    if isinstance(env, list):
      self._env = env
    else:
      self._env = [env]
    batch_size = batch_size or len(env)
    self._np_state = numpy_storage.NumpyState()
    self._np_state.adds_to_buff = np.array(0, dtype=float)
    # used so that buff is not over-populated by returned trajectories from short episodes
    super(CubeAverageScoreMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_last())

    if len(is_last[0]) > 0:
      for idx in is_last[0]:
        self.add_to_buffer([self._env[idx].gym.last_score])


@gin.configurable
class AverageGymInfoMetric(py_metrics.StreamingMetric):
  """Computes average score at end of trajectory"""
  def __init__(self, env, info_key='score', name='AverageScore', buffer_size=10,
               batch_size=None):
    """
    Creates an CubeAverageScoreMetric.
    Args:
      env: Instance of gym.Env that implements get_score() which updates the
           metric
      info_key: str of info dict key that is being averaged
      name: metric name
      buffer_size: number of episodes to compute average over
    """

    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    self._wrapped_gym_envs = env
    self._info_key = info_key
    batch_size = batch_size or len(env)
    self._np_state = numpy_storage.NumpyState()
    super(AverageGymInfoMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_last())

    if len(is_last[0]) > 0:
      for idx in is_last[0]:
        self.add_to_buffer([self._wrapped_gym_envs[idx].get_info().get(self._info_key)])


@gin.configurable
class ThreeFingerAverageSuccessMetric(py_metrics.StreamingMetric):
  """Computes average success rate for envs which terminate with positive reward in successful eps."""

  def __init__(self, name='AverageSuccess', buffer_size=10, batch_size=None):
    """Creates an AverageSuccessMetric."""
    # Set a dummy value on self._np_state.obs_val so it gets included in
    # the first checkpoint (before metric is first called).
    super(ThreeFingerAverageSuccessMetric, self).__init__(
        name, buffer_size=buffer_size, batch_size=batch_size)

  def _reset(self, batch_size):
    return

  def _batched_call(self, trajectory):
    """Processes the trajectory to update the metric.

    Args:
      trajectory: a tf_agents.trajectory.Trajectory.
    """

    is_last = np.where(trajectory.is_last())

    if len(is_last[0]) > 0:
      succ = np.logical_and(
          np.logical_not(trajectory.observation['task_agn_rew'][is_last]),
          trajectory.reward[is_last] > 1.)
      self.add_to_buffer(succ)