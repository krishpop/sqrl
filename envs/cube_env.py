import numpy as np
import gin
import gym

from pddm.envs.cube import cube_env

# rotate 20 deg about y axis (cos(a/2), sin(a/2), 0, 0) (up/down)
# rotate 20 deg about z axis (cos(a/2), 0, 0, sin(a/2)) (left/right)

GOAL_TASKS = {
      'left': [0, 0, -1.5],
      'right': [0, 0, 1.5],
      'up': [1.5, 0, 0],
      'down': [-1.5, 0, 0],

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


class SafemrlCubeEnv(cube_env.CubeEnv):

    def __init__(self, same_goals=False, goal_task=('left', 'right', 'up', 'down'),
                 max_steps=100):
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
      self._goal_options = [GOAL_TASKS[k] for k in goal_task]
      super(SafemrlCubeEnv, self).__init__()
      self._last_score = self.get_score(self.unwrapped.obs_dict)

    @property
    def last_score(self):
      return self._last_score

    def do_reset(self, reset_pose, reset_vel, reset_goal=None):
      obs = super(SafemrlCubeEnv, self).do_reset(reset_pose, reset_vel, reset_goal)
      self._last_score = self.get_score(self.unwrapped.obs_dict)
      return obs

    def step(self, a):
      # removes everything but score from output info
      a = np.array(a).squeeze()
      o, r, d, i = super(SafemrlCubeEnv, self).step(a)
      self._last_score = i['score']
      i = {'score': i['score']}
      return o, r, d, i

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
