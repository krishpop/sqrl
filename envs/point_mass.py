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

"""Goal-conditioned PointMassEnv implementation with well and drunk spider envs.

Implementation of point-mass environment with well and drunk spider layouts.
Includes goal-conditioned and time limit bonus wrappers.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import gin
import gym
import numpy as np
try:
  from tf_agents.environments import wrappers
  from tf_agents.trajectories import time_step as ts
  loaded_agents = True
except ImportError:
  loaded_agents = False
  print('did not load tf_agents')

# Implementation of PointMass environment from Benjamin Eyesenbach:
#    https://github.com/google-research/google-research/tree/master/sorb

WALLS = {
    'IndianWell':
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'IndianWell2':
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'IndianWell3':
        np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]),
    'DrunkSpider':
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
    'DrunkSpiderShort':
        np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]),
}


def resize_walls(walls, factor):
  """Increase the environment by rescaling.

  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment.

  Returns:
    walls: rescaled walls
  """
  if isinstance(factor, tuple):
    resize_w, resize_h = factor
  else:
    resize_w, resize_h = factor, factor
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(resize_h)])  # pylint: disable=g-complex-comprehension
  col_indices = np.array([i for i in range(width) for _ in range(resize_w)])  # pylint: disable=g-complex-comprehension
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (resize_h * height, resize_w * width)
  return walls


@gin.configurable
class PointMassEnv(gym.Env):
  """Class for 2D navigation in PointMass environment."""

  def __init__(self,
               env_name='DrunkSpiderShort',
               start=None,
               resize_factor=(1,1),
               action_noise=0.,
               action_scale=1.,
               start_bounds=None,
               alive_bonus=0.,
               action_pen=0.):
    """Initialize the point environment.

    Args:
      env_name: environment name
      start: starting position
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
      action_scale: (float) Scales action magnitude by a constant factor.
      start_bounds: starting bound
      alive_bonus: a live bonus
      action_pen: penlaty for taking actions
    """
    walls = env_name
    if start is None:
      self._start = start
    else:
      self._start = np.array([x + .5 for x in start], dtype=np.float32)
    self.state = self._start

    if resize_factor[0] > 1 or resize_factor[1] > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    (height, width) = self._walls.shape
    self._height = height
    self._width = width

    if start_bounds is not None:
      self._start_space = gym.spaces.Box(
          low=start_bounds[0], high=start_bounds[1], dtype=np.float32)
    else:
      self._start_space = gym.spaces.Box(
          low=np.zeros(2), high=np.array([height, width]), dtype=np.float32)

    self._action_noise = action_noise
    self._action_scale = action_scale
    self._alive_bonus = alive_bonus
    self._action_pen = float(action_pen)
    self.action_space = gym.spaces.Box(
        low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
    self.observation_space = gym.spaces.Dict({
        'observation':
            gym.spaces.Box(
                low=np.array([0.0, 0.0]),
                high=np.array([self._height, self._width]),
                dtype=np.float32),
        'task_agn_rew':  # whether or not agent has fallen
            gym.spaces.Box(low=np.array(0), high=np.array(1))
    })
    self._validate_start_goal()
    self.seed()
    self.reset()

  def _seed(self, seed=0):
    np.random.seed(seed)
    return [seed]

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def _validate_start_goal(self):
    if self._start is not None:
      assert not self._is_out_of_bounds(self._start), \
          'start must be in bounds of env'
      assert not self.is_fallen(self._start), 'start must not be in fallen state'

  def _sample_empty_state(self):
    if self._start is not None:
      state = self._start.copy()
    else:
      state = self._start_space.sample()
    assert not self.is_fallen(state)
    return state

  def _is_out_of_bounds(self, state):
    return not self.observation_space['observation'].contains(state)

  def is_fallen(self, state=None):
    if state is None:
      state = self.state
    if self._is_out_of_bounds(state):
      return True
    (i, j) = self._discretize_state(state)
    return self._walls[i, j] == 1

  def reset(self, reset_args=None):
    self.state = reset_args or self._sample_empty_state()
    obs = dict(observation=self.state.copy(), task_agn_rew=0.)
    return obs

  def step(self, action):
    if len(action.shape) == 2:
      action = action[0]
    action = self._action_scale * action
    if self._action_noise > 0:
      action += (np.random.sample((2,)) - 0.5) * self._action_noise
    action = np.clip(action, self.action_space.low * self._action_scale, self.action_space.high * self._action_scale)
    num_substeps = 10
    fallen, rew = False, 0.

    dt = 1.0 / num_substeps
    num_axis = len(action)

    for _ in range(num_substeps):
      new_state = self.state.copy()
      for axis in range(num_axis):
        new_state[axis] += dt * action[axis]
        if self.is_fallen(new_state):
          rew = -1. * self._is_out_of_bounds(new_state)
          new_state = np.clip(new_state, self.observation_space['observation'].low,
                              self.observation_space['observation'].high)
          fallen = True
      self.state = new_state
      if fallen:
        break
    # control rew discourages large actions, in range [-1.414*c, 0]
    rew += -np.linalg.norm(action) * self._action_pen
    obs = dict(
        observation=self.state.copy(), task_agn_rew=float(fallen))
    return obs, rew + self._alive_bonus, fallen, {}

  @property
  def walls(self):
    return self._walls


@gin.configurable
class PointMassAcNoiseEnv(PointMassEnv):
  def __init__(self, n_tasks=10, task_bounds=(0., 0.4), domain_rand=False, *env_args, **env_kwargs):
    ac_noise_low, ac_noise_high = task_bounds
    self._tasks = np.linspace(ac_noise_low, ac_noise_high, n_tasks)
    ac_noise = self._tasks[np.random.randint(n_tasks)]
    if env_kwargs is None:
      env_kwargs = {}
    env_kwargs['action_noise'] = ac_noise
    self._domain_rand = domain_rand
    super(PointMassAcNoiseEnv, self).__init__(*env_args, **env_kwargs)

  def reset_task(self, task_idx=None):
    if task_idx is None:
      task_idx = np.random.randint(len(self._tasks))
    self._action_noise = self._tasks[task_idx]

  def reset(self):
    if self._domain_rand:
      self.reset_task()
    return super(PointMassAcNoiseEnv, self).reset()


@gin.configurable
class PointMassAcScaleEnv(PointMassEnv):
  def __init__(self, n_tasks=16, task_bounds=(0.5, 2.), domain_rand=False, *env_args, **env_kwargs):
    ac_scale_low, ac_scale_high = task_bounds
    self._tasks = np.linspace(ac_scale_low, ac_scale_high, n_tasks)
    ac_scale = self._tasks[np.random.randint(n_tasks)]
    if env_kwargs is None:
      env_kwargs = {}
    env_kwargs['action_scale'] = ac_scale
    self._domain_rand = domain_rand
    super(PointMassAcScaleEnv, self).__init__(*env_args, **env_kwargs)

  def reset_task(self, task_idx=None):
    if task_idx is None:
      task_idx = np.random.randint(len(self._tasks))
    self._action_scale = self._tasks[task_idx]

  def reset(self):
    if self._domain_rand:
      self.reset_task()
    return super(PointMassAcScaleEnv, self).reset()


@gin.configurable
class GoalConditionedPointWrapper(gym.Wrapper):
  """Wrapper that appends goal to state produced by environment."""

  def __init__(self,
               env,
               goal=None,
               normalize_obs=False,
               task_rew_type='l2',
               reset_on_fall=True,
               goal_bounds=None,
               threshold_distance=1.0,
               fall_penalty=0.,
               max_episode_steps=30):
    """Initialize the environment.

    Args:
      env: an environment.
      goal: default goal to use, instead of sampling
      normalize_obs: returns normalized observations
      task_rew_type: string in ('l1', '-1', 'l2') indicating task reward to use
      reset_on_fall: boolean indicating if episode finishes when fallen == True
      goal_bounds: bounds for goal
      threshold_distance: (float) States are considered equivalent if they are
        at most this far away from one another.
      fall_penalty: penalty for falls
    """
    self._default_goal = goal if goal is None else np.array([x + .5 for x in goal], dtype=np.float32)
    self._task_rew_type = task_rew_type
    self._reset_on_fall = reset_on_fall
    self._threshold_distance = threshold_distance
    self._fall_penalty = fall_penalty
    self._max_episode_steps = max_episode_steps
    self._steps_taken = 0
    self._norm_obs = normalize_obs
    self._seed = env._seed

    if normalize_obs:
      obs_space = gym.spaces.Box(
          low=np.array([0., 0.]), high=np.array([1., 1.]), dtype=np.float32)
    else:
      obs_space = env.observation_space['observation']
    if goal_bounds is not None:
      goal_space = gym.spaces.Box(
          low=np.array(goal_bounds[0]), high=np.array(goal_bounds[1]), dtype=np.float32)
    else:
      goal_space = env.observation_space['observation']
    if goal:
      assert goal_space.contains(np.array(goal)), 'goal: {} not in goal space: {}'.format(goal, goal_bounds)
    super(GoalConditionedPointWrapper, self).__init__(env)

    # overwrites observation space to include goals
    self.observation_space = gym.spaces.Dict({
        'observation': obs_space,
        'goal': goal_space,
        'task_agn_rew': env.observation_space['task_agn_rew']
    })
    self.reset()  # sets up goal value

  def _normalize_obs(self, obs):
    return np.array(
        [obs[0] / float(self.env._height), obs[1] / float(self.env._width)])  # pylint: disable=protected-access

  def _sample_goal(self, obs):
    goal_dist = 0
    state = obs['observation']
    while goal_dist < 4:
      goal = self.observation_space['goal'].sample()
      goal_dist = np.abs(state - goal).sum()
    return (obs, goal)

  def _compute_reward(self, observation=None, goal=None):
    task_rew = 0.
    if observation is None:
      observation = self.state
    if goal is None:
      goal = self._goal

    if self._is_done(self.state, self.goal):
      task_rew = 30.
    elif self._task_rew_type == 'l1':
      # task_rew range: [-1, 0]
      max_dist = self.env._height + self.env._width
      task_rew += -np.abs(observation - goal).sum() / max_dist
    elif self._task_rew_type == 'l2':
      max_dist = np.sqrt(self.env._height ** 2 + self.env._width ** 2)
      task_rew += -np.linalg.norm(observation - goal) / max_dist
    elif self._task_rew_type == '+l2':  # positive l2
      max_dist = np.sqrt(self.env._height ** 2 + self.env._width ** 2)
      task_rew += 1 - np.linalg.norm(observation - goal) / max_dist
    elif self._task_rew_type == '+l1':
      max_dist = self.env._height + self.env._width
      task_rew += 1 - np.abs(observation - goal).sum() / max_dist
    elif self._task_rew_type == '-1':  # alive penalty
      task_rew += -.1
    return task_rew

  def _is_done(self, obs, goal):
    """Determines whether goal reached."""
    return np.linalg.norm(obs - goal) < self._threshold_distance

  @property
  def goal(self):
    return self._goal

  def reset(self):
    """Resets environment, sampling goal if self._sample_goal == True."""
    self._steps_taken = 0
    obs = self.env.reset()
    if self._default_goal is not None:
      (obs, goal) = (obs, self._default_goal.copy())
    else:
      (obs, goal) = self._sample_goal(obs)

    obs['goal'] = self._goal = goal
    if self._norm_obs:
      obs['observation'] = self._normalize_obs(obs['observation'])
    return obs

  def step(self, action):
    self._steps_taken += 1
    assert self._steps_taken <= self._max_episode_steps, "Did not call reset after {} steps".format(self._steps_taken)

    obs, rew, done, _ = self.env.step(action)
    obs['goal'] = self.goal
    if self._norm_obs:
      obs['observation'] = self._normalize_obs(obs['observation'])
    self.state = obs['observation']
    if done and self._reset_on_fall:
      return obs, self._fall_penalty, done, {}

    task_rew = self._compute_reward()

    return obs, task_rew, self._is_done(self.state, self.goal), {}


class SafetyGymWrapper(gym.Wrapper):
  def __init__(self, env, fall_cost=1.):
    super().__init__(env)
    self.observation_space = env.observation_space['observation']
    self._fall_cost = fall_cost

  def reset(self):
    obs = super().reset()
    return obs['observation']

  def step(self, action):
    o, r, d, i = super().step(action)
    i.update({'cost': o['task_agn_rew'] * self._fall_cost})
    o = o['observation']
    return o, r, d, i


if loaded_agents:
  class NonTerminatingTimeLimit(wrappers.PyEnvironmentBaseWrapper):
    """Resets the environment without setting done = True.

    Resets the environment if either these conditions holds:
      1. The base environment returns done = True
      2. The time limit is exceeded.
    """

    def __init__(self, env, duration):
      super(NonTerminatingTimeLimit, self).__init__(env)
      self._duration = duration
      self._step_count = None

    def _reset(self):
      self._step_count = 0
      return self._env.reset()

    @property
    def duration(self):
      return self._duration

    def _step(self, action):
      if self._step_count is None:
        return self.reset()

      timestep = self._env.step(action)  # pylint: disable=protected-access

      self._step_count += 1
      if self._step_count >= self._duration or timestep.is_last():
        self._step_count = None

      return timestep


  @gin.configurable
  class TimeLimitBonus(wrappers.PyEnvironmentBaseWrapper):
    """End episodes after specified steps, adding early bonus/penalty."""

    def __init__(self, env, max_episode_steps, early_term_bonus=1., early_term_penalty=-1.,
                 time_limit_penalty=-30.):
      super(TimeLimitBonus, self).__init__(env)
      self._duration = max_episode_steps
      self._num_steps = None
      self._early_term_bonus = early_term_bonus
      self._early_term_penalty = -abs(early_term_penalty)
      self._time_limit_penalty = -abs(time_limit_penalty)

    def _reset(self):
      self._num_steps = 0
      return self._env.reset()

    def _step(self, action):
      if self._num_steps is None:
        return self.reset()

      time_step = self._env.step(action)

      self._num_steps += 1
      reached_time_limit = False
      if self._num_steps >= self._duration:
        time_step = time_step._replace(step_type=ts.StepType.LAST)
        reached_time_limit = True

      if time_step.is_last():  # if episode terminated
        reward = time_step.reward
        if not time_step.observation['task_agn_rew']:  # and agent hasn't fallen
          if not reached_time_limit:
            reward = (time_step.reward + self._early_term_bonus *
                      (self._duration - self._num_steps))  # add early termination bonus (if duration < num_steps
          elif reached_time_limit and self._time_limit_penalty:
            reward = time_step.reward + self._time_limit_penalty
        else:
          reward = (time_step.reward + self._early_term_penalty * (not reached_time_limit) *
                    (self._duration - self._num_steps + 1))

        time_step = time_step._replace(
            reward=reward.astype(time_step.reward.dtype))

        self._num_steps = None

      return time_step

    @property
    def duration(self):
      return self._duration


@gin.configurable
def env_load_fn(environment_name='DrunkSpiderShort',
                max_episode_steps=50,
                resize_factor=(1,1),
                terminate_on_timeout=True,
                start=(0, 3), goal=(7,3), goal_bounds=[(6,2), (7,4)],
                fall_penalty=0.,
                reset_on_fall=False,
                gym_env_wrappers=[],
                gym=False):
  """Loads the selected environment and wraps it with the specified wrappers.

  Args:
    environment_name: Name for the environment to load.
    max_episode_steps: If None the max_episode_steps will be set to the default
      step limit defined in the environment's spec. No limit is applied if set
      to 0 or if there is no timestep_limit set in the environment's spec.
    resize_factor: A factor for resizing.
    terminate_on_timeout: Whether to set done = True when the max episode steps
      is reached.

  Returns:
    A PyEnvironmentBase instance.
  """
  if resize_factor != (1, 1):
    if start:
      start = (start[0] * resize_factor[0], start[1] * resize_factor[1])
    if goal:
      goal = (goal[0] * resize_factor[0], goal[1] * resize_factor[1])
    if goal_bounds:
      goal_bounds = [(g[0] * resize_factor[0], g[1] * resize_factor[1]) for g in goal_bounds]

  if 'acnoise' in environment_name.split('-'):
    environment_name = environment_name.split('-')[0]
    gym_env = PointMassAcNoiseEnv(
      start=start, env_name=environment_name, resize_factor=resize_factor)
  elif 'acscale' in environment_name.split('-'):
    environment_name = environment_name.split('-')[0]
    gym_env = PointMassAcScaleEnv(
      start=start, env_name=environment_name, resize_factor=resize_factor)
  else:
    gym_env = PointMassEnv(
        start=start, env_name=environment_name, resize_factor=resize_factor)

  gym_env = GoalConditionedPointWrapper(gym_env, goal=goal, goal_bounds=goal_bounds, fall_penalty=-abs(fall_penalty),
                                        reset_on_fall=reset_on_fall, max_episode_steps=max_episode_steps)
  for wrapper in gym_env_wrappers:
    gym_env = wrapper(gym_env)
  if gym:
    return gym_env

  from tf_agents.environments import gym_wrapper
  env = gym_wrapper.GymWrapper(
      gym_env, discount=1.0, auto_reset=True, simplify_box_bounds=False)

  if max_episode_steps > 0:
    if terminate_on_timeout:
      env = TimeLimitBonus(env, max_episode_steps)
    else:
      env = NonTerminatingTimeLimit(env, max_episode_steps)

  return env


class PointMassObservationWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.observation_space = env.observation_space['observation']

  def observation(self, obs):
    return obs['observation']

  def step(self, action):
      o, r, d, i = self.env.step(action)
      i.update(o)
      o = self.observation(o)
      return o, r, d, i
