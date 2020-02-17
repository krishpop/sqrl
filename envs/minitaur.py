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
"""Custom Minitaur environment with target velocity.

Implements minitaur environment with rewards dependent on closeness to goal
velocity. Extends the MinitaurExtendedEnv class from PyBullet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import gym
import numpy as np

from pybullet_envs.minitaur.envs import minitaur_extended_env
from safemrl.utils import misc

ENV_DEFAULTS = {
  "accurate_motor_model_enabled": True,
  "never_terminate": False,
  "history_length": 5,
  "urdf_version": "rainbow_dash_v0",
  "history_include_actions": True,
  "control_time_step": 0.02,
  "history_include_states": True,
  "include_leg_model": True
}

@gin.configurable
class MinitaurGoalVelocityEnv(minitaur_extended_env.MinitaurExtendedEnv):
  """The 'extended' minitaur env with a target velocity."""

  def __init__(self,
               goal_vel=0.3,
               goal_limit=0.8,
               max_steps=500,
               debug=False,
               butterworth=False,
               friction=None,
               **kwargs):
    self.set_sample_goal_args(goal_limit, goal_vel)
    self._current_vel = 0.
    self._debug = debug
    self._max_steps = max_steps
    self._butterworth = butterworth
    if not kwargs:
      kwargs = ENV_DEFAULTS
    super(MinitaurGoalVelocityEnv, self).__init__(**kwargs)
    self.set_foot_friction(friction)

  @property
  def current_vel(self):
    return self._current_vel

  def _termination(self):
    """Determines whether the env is terminated or not.

    Checks whether 1) the front leg is bent too much 2) the time exceeds
    the manually set weights or 3) if the minitaur has "fallen"
    Returns:
      terminal: the terminal flag whether the env is terminated or not
    """
    if self._never_terminate:
      return False

    if self._counter >= self._max_steps:
      return True

    return self.is_fallen()  # terminates automatically when in fallen state

  def is_fallen(self):
    if super(MinitaurGoalVelocityEnv, self).is_fallen():
      return True
    leg_model = self.convert_to_leg_model(self.minitaur.GetMotorAngles())
    swing0 = leg_model[0]
    swing1 = leg_model[2]
    maximum_swing_angle = 0.8
    if swing0 > maximum_swing_angle or swing1 > maximum_swing_angle:
      return True
    return False

  def set_foot_friction(self, friction=None):
    self._foot_friction = friction
    if friction:
      self.minitaur.SetFootFriction(friction)

  def set_sample_goal_args(self, goal_limit=None, goal_vel=None):
    if goal_limit is not None:
      self._goal_limit = goal_limit
    if goal_vel is not None:
      self._goal_vel = goal_vel

  def reset(self, **kwargs):
    if kwargs.get('initial_motor_angles', None):
      return super(minitaur_extended_env.MinitaurExtendedEnv, self).reset(**kwargs)
    return super(MinitaurGoalVelocityEnv, self).reset()

  def reward(self):
    """Compute rewards for the given time step.

    It considers two terms: 1) forward velocity reward and 2) action
    acceleration penalty.
    Returns:
      reward: the computed reward.
    """
    current_base_position = self.minitaur.GetBasePosition()
    dt = self.control_time_step
    self._current_vel = velocity = (current_base_position[0] - self._last_base_position[0]) / dt
    vel_clip = np.clip(velocity, -self._goal_limit, self._goal_limit)
    velocity_reward = self._goal_vel - np.abs(self._goal_vel - vel_clip)

    action = self._past_actions[self._counter - 1]
    prev_action = self._past_actions[max(self._counter - 2, 0)]
    prev_prev_action = self._past_actions[max(self._counter - 3, 0)]
    acc = action - 2 * prev_action + prev_prev_action
    action_acceleration_penalty = np.mean(np.abs(acc))

    reward = 0.0
    reward += 1.0 * velocity_reward
    reward -= 0.01 * action_acceleration_penalty  # TODO: lowering acceleration penalty, try 0.01, 0.002

    if self._debug:
      self.pybullet_client.addUserDebugText('Current velocity: {:3.2f}'.format(
        self._current_vel), [0, 0, 1], [1, 0, 0])
    return reward

  def step(self, action):
    if self._butterworth:
      action *= self._action_scale
      self._past_actions[self._counter] = action
      data = self._past_actions[:self._counter+1]
      action = np.apply_along_axis(misc.butter_bandpass_filter, 0, data)
      action /= self._action_scale
    return super(MinitaurGoalVelocityEnv, self).step(action)


@gin.configurable
class TaskAgnWrapper(gym.Wrapper):
  def __init__(self, env):
    super(TaskAgnWrapper, self).__init__(env)
    if isinstance(self.observation_space, gym.spaces.Dict):
       # add task_agn_rew to observation space dict, instead of nesting
      spaces = list(self.observation_space.spaces.items())
      spaces.append(('task_agn_rew', gym.spaces.Box(0, 1, shape=())))
      self.observation_space = gym.spaces.Dict(dict(spaces))
    else:
      self.observation_space = gym.spaces.Dict({
        'observation': self.observation_space,
        'task_agn_rew': gym.spaces.Box(0, 1, shape=())
      })

  def step(self, action):
    o, r, d, i = self.env.step(action)
    if isinstance(o, dict):
      o['task_agn_rew'] = 0.
    else:
      o = {'observation': o, 'task_agn_rew': 0.}
    if d and self.unwrapped.is_fallen():
      o['task_agn_rew'] = 1.
    return o, r, d, i

  def reset(self, **kwargs):
    o = self.env.reset(**kwargs)
    if isinstance(o, dict):
      o['task_agn_rew'] = 0.
    else:
      o = {'observation': o, 'task_agn_rew': 0.}
    return o


@gin.configurable
class CurrentVelWrapper(gym.Wrapper):
  def __init__(self, env):
    super(CurrentVelWrapper, self).__init__(env)
    if isinstance(self.observation_space, gym.spaces.Dict):
       # add task_agn_rew to observation space dict, instead of nesting
      spaces = list(self.observation_space.spaces.items())
      spaces.append(('current_vel', gym.spaces.Box(-np.inf, np.inf, shape=())))
      self.observation_space = gym.spaces.Dict(dict(spaces))
    else:
      self.observation_space = gym.spaces.Dict({
        'observation': self.observation_space,
        'current_vel': gym.spaces.Box(-np.inf, np.inf, shape=())
      })

  def step(self, action):
    o, r, d, i = self.env.step(action)
    if isinstance(o, dict):
      o['current_vel'] = self.unwrapped.current_vel
    else:
      o = {'observation': o, 'current_vel': self.unwrapped.current_vel}
    return o, r, d, i

  def reset(self, **kwargs):
    o = self.env.reset(**kwargs)
    if isinstance(o, dict):
      o['current_vel'] = self.unwrapped.current_vel
    else:
      o = {'observation': o, 'current_vel': self.unwrapped.current_vel}
    return o