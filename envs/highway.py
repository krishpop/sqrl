import gin
import gym
import numpy as np
from gym import spaces


@gin.configurable
class ContAcWrapper(gym.ActionWrapper):
  def __init__(self, env, *args, **kwargs):
    assert isinstance(env.action_space, spaces.Discrete), ('ContAcWrapper only works for envs '
                                                           'with discrete action spaces')
    super(ContAcWrapper, self).__init__(env, *args, **kwargs)
    self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
    self._n_discrete = self.env.action_space.n
    self._buckets = np.linspace(0., 1., self._n_discrete + 1)

  def action(self, action):
    disc_ac = np.digitize(action, self._buckets, right=False).item()
    return np.clip(disc_ac - 1, 0., 1.)  # returns discrete action~

@gin.configurable
class TaskAgnWrapper(gym.Wrapper):
  def __init__(self, env):
    super(TaskAgnWrapper, self).__init__(env)
    self.observation_space = spaces.Dict({'observation': self.observation_space,
                                          'task_agn_rew': spaces.Box(0., 1., shape=())})

  def step(self, action):
    o, r, d, i = self.env.step(action)
    o = o.reshape(-1)
    o = {'observation': o, 'task_agn_rew': float(i['crashed'])}
    return o, r, d, i

  def reset(self):
    o = self.env.reset()
    o = {'observation': o, 'task_agn_rew': 0.}
    return o
