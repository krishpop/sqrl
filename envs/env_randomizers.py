import random
import numpy as np
import gin
from pybullet_envs.minitaur.envs import env_randomizer_base


@gin.configurable
class MinitaurFootFrictionEnvRandomizer(env_randomizer_base.EnvRandomizerBase):
  def __init__(self, minitaur_leg_friction_range=(0.5, 1.25)):
    self._minitaur_leg_friction_range = minitaur_leg_friction_range

  def randomize_env(self, env):
    self._randomize_minitaur(env.minitaur)

  def _randomize_minitaur(self, minitaur):
    randomized_foot_friction = random.uniform(self._minitaur_leg_friction_range[0],
                                              self._minitaur_leg_friction_range[1])
    minitaur.SetFootFriction(randomized_foot_friction)


@gin.configurable
class MinitaurFootFrictionEnvTaskRandomizer(env_randomizer_base.EnvRandomizerBase):
  def __init__(self, minitaur_leg_friction_range=(0.5, 1.25),
               n_tasks=100, n_train_tasks=80, seed=1, train=True):
    """Splits friction range into train and test tasks"""
    self._n_tasks = n_tasks
    self._rs = np.random.RandomState(seed)
    tasks = self._rs.uniform(minitaur_leg_friction_range[0], minitaur_leg_friction_range[1], n_tasks)
    self._minitaur_leg_friction_range = minitaur_leg_friction_range
    self._train_leg_friction_tasks = tasks[:n_train_tasks]
    self._test_leg_friction_tasks = tasks[n_train_tasks:]
    self._training = train

  def randomize_env(self, env):
    self._randomize_minitaur(env.minitaur)

  def _randomize_minitaur(self, minitaur):
    if self._training:
      n_train = len(self._train_leg_friction_tasks)
      friction = self._train_leg_friction_tasks[self._rs.randint(n_train)]
    else:
      n_test = len(self._test_leg_friction_tasks)
      friction = self._test_leg_friction_tasks[self._rs.randint(n_test)]
    minitaur.SetFootFriction(friction)
