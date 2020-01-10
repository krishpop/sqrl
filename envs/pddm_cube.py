import numpy as np
import gin
import gym

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
