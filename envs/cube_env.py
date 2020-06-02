import collections

import numpy as np
import gin
import gym
import tensorflow as tf

from pddm.envs.cube import cube_env

# from tf_agents.specs import array_spec
# from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper

# rotate 20 deg about y axis (cos(a/2), sin(a/2), 0, 0) (up/down)
# rotate 20 deg about z axis (cos(a/2), 0, 0, sin(a/2)) (left/right)

GOAL_TASKS = {
      'left': [0, 0, -1.5],
      'right': [0, 0, 1.5],
      'up': [1.5, 0, 0],
      'down': [-1.5, 0, 0],

      'more_left': [0, 0, -2.2],
      'more_right': [0, 0, 2.2],
      'more_up': [2.2, 0, 0],
      'more_down': [-2.2, 0, 0],

      'half_up': [0.7, 0, 0],
      'half_down': [-0.7, 0, 0],
      'half_left': [0, 0, -0.7],
      'half_right': [0, 0, 0.7],

      'slight_up': [0.35, 0, 0],
      'slight_down': [-0.35, 0, 0],
      'slight_left': [0, 0, -0.35],
      'slight_right': [0, 0, 0.35],
}

#####################################
#####################################


@gin.configurable
class SafemrlCubeEnv(cube_env.CubeEnv):

    def __init__(self, same_goals=False, goal_task=('left', 'right', 'up', 'down'),
                 drop_penalty=-1000., max_steps=100, action_history=7):
      #####################################
      #####################################

      # CHOOSE one of these goal options here:
      # goal_options = [half_up, half_down, half_left, half_right, slight_right, slight_left, slight_down, slight_up, left, right, up, down]
      # goal_options = [half_up, half_down, half_left, half_right]
      # goal_options = [half_up, half_down, half_left, half_right, slight_right, slight_left, slight_down, slight_up]
      # goal_options = [left, right]
      # goal_options = [up, down]

      #####################################
      #####################################
      self._max_steps = max_steps
      self._same_goals = same_goals
      self._drop_penalty = drop_penalty
      self._goal_options = [GOAL_TASKS[k] for k in goal_task]
      self._action_history = action_history
      super(SafemrlCubeEnv, self).__init__()
      if action_history:
        ac_shape = np.prod(self._last_actions.shape)
        ac_low = np.repeat(self.action_space.low, action_history)
        ac_high = np.repeat(self.action_space.high, action_history)
        obs_low = np.concatenate([self.observation_space.low[:ac_shape], ac_low])
        obs_high = np.concatenate([self.observation_space.high[:ac_shape], ac_high])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high,
                                                dtype=self.observation_space.dtype)
      self._last_score = self.get_score(self.unwrapped.obs_dict)

    @property
    def last_score(self):
      return self._last_score

    def set_sample_goal_args(self, goal_task=[]):
      self._goal_options = [GOAL_TASKS[k] for k in goal_task]

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):
      obs = super(SafemrlCubeEnv, self).do_reset(reset_pose, reset_vel, reset_goal)
      self._last_score = self.get_score(self.unwrapped.obs_dict)
      return obs

    def _get_obs(self):
      o = super(SafemrlCubeEnv, self)._get_obs()
      if self._action_history:
        o = np.concatenate([o, self._last_actions.T.flatten()])
      return o

    def step(self, a):
      # removes everything but score from output info
      a = np.array(a).squeeze()
      if self.startup:
        self._last_actions = np.zeros((self.n_jnt, self._action_history))
      elif self._action_history:
        self._last_actions = np.roll(self._last_actions, -1, axis=-1)
        self._last_actions[:, -1] = a
      o, r, d, i = super(SafemrlCubeEnv, self).step(a)
      self._last_score = i['score']
      i = {'score': i['score']}
      return o, r, d, i

    def get_reward(self, observations, actions):
        #initialize and reshape as needed, for batch mode
        r_total, dones = super(SafemrlCubeEnv, self).get_reward(observations, actions)

        if len(observations.shape)==1:
            observations = np.expand_dims(observations, axis = 0)
            actions = np.expand_dims(actions, axis = 0)
            batch_mode = False
        else:
            batch_mode = True

        obj_height = observations[:,24+2]
        zeros = np.zeros(obj_height.shape)

        #fall
        is_fall = zeros.copy()
        is_fall[obj_height < -0.1] = 1

        #done based on is_fall
        dones = (is_fall==1) if not self.startup else zeros

        #rewards
        self.reward_dict['drop_penalty'] = self._drop_penalty * is_fall
        self.reward_dict['r_total'] = self.reward_dict['ori_dist'] + self.reward_dict['drop_penalty']

        if not batch_mode:
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones

    def create_goal_trajectory(self):

      len_of_goals = self._max_steps

      # A single rollout consists of alternating between 2 (same or diff) goals:
      if self._same_goals:
        goal_selected1 = np.random.randint(len(self._goal_options))
        goal_selected2 = goal_selected1
      else:
        goal_selected1 = np.random.randint(len(self._goal_options))
        goal_selected2 = np.random.randint(len(self._goal_options))
      goals = [self._goal_options[goal_selected1], self._goal_options[goal_selected2]]

      # Create list of these goals
      time_per_goal = 35
      step_num = 0
      curr_goal_num = 0
      goal_traj = []
      while step_num < len_of_goals:
        goal_traj.append(np.tile(goals[curr_goal_num], (time_per_goal, 1)))
        if curr_goal_num == 0:
          curr_goal_num = 1
        else:
          curr_goal_num = 0
        step_num += time_per_goal

      goal_traj = np.concatenate(goal_traj)
      return goal_traj


@gin.configurable
class CubeTaskAgnWrapper(gym.Wrapper):
  def __init__(self, env):
    super(CubeTaskAgnWrapper, self).__init__(env)
    self.observation_space = gym.spaces.Dict({
      'observation': self.observation_space,
      'task_agn_rew': gym.spaces.Box(np.array(0), np.array(1))
    })

  def step(self, action):
    o, r, d, i = super(CubeTaskAgnWrapper, self).step(action)
    o_dict = {'observation': o, 'task_agn_rew': 0.}
    if d and self.env.reward_dict.get('drop_penalty', 0) != 0:
      o_dict['task_agn_rew'] = 1.
    return o_dict, r, d, i

  def reset(self, **kwargs):
    o = super(CubeTaskAgnWrapper, self).reset(**kwargs)
    return {'observation': o, 'task_agn_rew': 0.}


# @gin.configurable
# class ActionHistoryWrapper(PyEnvironmentBaseWrapper):
#   """Adds observation and action history to the environment's observations."""
#
#   def __init__(self, env, history_length=7):
#     """Initializes a HistoryWrapper.
#
#     Args:
#       env: Environment to wrap.
#       history_length: Length of the history to attach.
#       include_actions: Whether actions should be included in the history.
#     """
#     super(ActionHistoryWrapper, self).__init__(env)
#     self._history_length = history_length
#
#     self._zero_action = self._zeros_from_spec(env.action_spec())
#
#     self._action_history = collections.deque(maxlen=history_length)
#
#     self._observation_spec = self._get_observation_spec()
#
#   def _get_observation_spec(self):
#
#     def _update_shape(spec):
#       return array_spec.update_spec_shape(spec,
#                                           (self._history_length,) + spec.shape)
#
#     observation_spec = self._env.observation_spec()
#
#     action_spec = tf.nest.map_structure(_update_shape,
#                                         self._env.action_spec())
#     flattened_shape = sum(np.prod(obs.shape) for obs in self._flatten_nested_observation(
#       [observation_spec['observation'], action_spec]))
#     return array_spec.ArraySpec(shape=flattened_shape, dtype=observation_spec.dtype,
#                                 name='packed_observations')
#
#   def observation_spec(self):
#     return self._observation_spec
#
#   def _zeros_from_spec(self, spec):
#
#     def _zeros(spec):
#       return np.zeros(spec.shape, dtype=spec.dtype)
#
#     return tf.nest.map_structure(_zeros, spec)
#
#   def _add_history(self, time_step, action):
#     self._action_history.append(action)
#
#     observation = {
#         'observation': time_step.observation,
#         'action': np.stack(self._action_history)
#     }
#
#     return time_step._replace(observation=observation)
#
#   def _reset(self):
#     self._action_history.extend([self._zero_action] *
#                                 (self._history_length - 1))
#
#     time_step = self._env.reset()
#     return self._add_history(time_step, self._zero_action)
#
#   def _step(self, action):
#     if self.current_time_step() is None or self.current_time_step().is_last():
#       return self._reset()
#
#     time_step = self._env.step(action)
#     return self._add_history(time_step, action)


class SafetyGymWrapper(gym.Wrapper):
  def __init__(self, env, fall_cost=1.):
    super(SafetyGymWrapper, self).__init__(env)
    self._fall_cost = fall_cost

  def step(self, action):
    o, r, d, i = super(SafetyGymWrapper, self).step(action)
    if d and self.env.reward_dict.get('drop_penalty', 0) != 0:
      i.update({'cost': self._fall_cost})
    else:
      i.update({'cost': 0})
    return o, r, d, i
