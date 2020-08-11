import gin
import gym
import numpy as np

from three_finger.envs.raw_controller_env import Gripper2DSamplePoseEnv


@gin.configurable
class ThreeFingerRawEnv(Gripper2DSamplePoseEnv):
  def __init__(self, reset_on_drop=True, max_steps=100, reward_type='contacts'):
    super().__init__(reset_on_drop=reset_on_drop, reward_type=reward_type)
    self.max_episode_steps = max_steps


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
    if i['dropped']:
      o['task_agn_rew'] = 1.
      r -= 50
    else:
      r += 6
    return o, r, d, i

  def reset(self, **kwargs):
    o = self.env.reset(**kwargs)
    if isinstance(o, dict):
      o['task_agn_rew'] = 0.
    else:
      o = {'observation': o, 'task_agn_rew': 0.}
    return o