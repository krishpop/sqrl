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

"""Register default envs."""

from __future__ import absolute_import

import gym

from safemrl.envs import minitaur, env_randomizers
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_terrain_randomizer
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_env_randomizer

try:
  from safemrl.envs import cube_env
except ImportError:
  print('cube_env was not imported')
  cube_env = None

try:
  from safemrl.envs import three_finger
except ImportError:
  print("three_finger was not imported")
  three_finger = None


from safemrl.envs import point_mass
from safemrl.envs import sg_envs
from gym.envs.registration import register

registered = "MinitaurGoalVelocityEnv-v0" in gym.envs.registry.env_specs

POINT_MASS_EPLEN = 30

if not registered:
  register(
    id="DrunkSpiderPointMassEnv-v0",
    entry_point=point_mass.env_load_fn,
    max_episode_steps=POINT_MASS_EPLEN,
    kwargs={"environment_name": "DrunkSpiderShort",
            "max_episode_steps": POINT_MASS_EPLEN, "resize_factor": (1, 1),
            "gym": True, "reset_on_fall": False}
  )

  register(
    id="DrunkSpiderPointMassEnv-v1",
    entry_point=point_mass.env_load_fn,
    max_episode_steps=POINT_MASS_EPLEN,
    kwargs={"environment_name": "DrunkSpiderShort",
            "max_episode_steps": POINT_MASS_EPLEN, "resize_factor": (1, 1),
            "gym": True, "reset_on_fall": True}
  )

  register(
    id="Safexp-DrunkSpiderPointMassGoal-v0",
    entry_point=point_mass.env_load_fn,
    max_episode_steps=POINT_MASS_EPLEN,
    kwargs={"environment_name": "DrunkSpiderShort",
            "max_episode_steps": POINT_MASS_EPLEN, "resize_factor": (1, 1),
            "gym": True, "gym_env_wrappers": [point_mass.SafetyGymWrapper],
            "reset_on_fall": True}
  )

  # MINITAUR ENVS
  register(
    id="MinitaurGoalVelocityEnv-v0",
    entry_point=minitaur.MinitaurGoalVelocityEnv,
    max_episode_steps=500,
    kwargs={'max_steps': 500}  # TODO: environment also takes max_steps as arg
  )

  randomizers = []
  randomizers.append(env_randomizers.MinitaurFootFrictionEnvRandomizer())
  register(
    id="MinitaurRandFrictionGoalVelocityEnv-v0",
    entry_point=minitaur.MinitaurGoalVelocityEnv,
    max_episode_steps=500,
    kwargs={'env_randomizer': randomizers, 'max_steps': 500}
  )

  randomizers = []
  randomizers.append(env_randomizers.MinitaurFootFrictionEnvTaskRandomizer())
  register(
    id="MinitaurRandFrictionGoalVelocityEnv-v1",
    entry_point=minitaur.MinitaurGoalVelocityEnv,
    max_episode_steps=500,
    kwargs={'env_randomizer': randomizers, 'max_steps': 500}
  )

  randomizers = []
  randomizers.append(minitaur_terrain_randomizer.MinitaurTerrainRandomizer())
  register(
    id="MinitaurRandTerrainGoalVelocityEnv-v0",
    entry_point=minitaur.MinitaurGoalVelocityEnv,
    max_episode_steps=500,
    kwargs={'env_randomizer': randomizers, 'max_steps': 500}
  )

  randomizers = [minitaur_env_randomizer.MinitaurEnvRandomizer()]
  register(
    id="MinitaurPhysRandEnv-v0",
    entry_point=minitaur.MinitaurGoalVelocityEnv,
    max_episode_steps=500,
    kwargs={'env_randomizer': randomizers, 'max_steps': 500}
  )

  # CUBE ENVS
  if cube_env:
    v_num = 0
    for ac_hist in [0, 7]:
      for n_steps in [500, 100, 1000]:
        for same_goals in [True, False]:
          register(
            id="SafemrlCube-v{}".format(v_num),
            entry_point=cube_env.SafemrlCubeEnv,
            max_episode_steps=n_steps,
            kwargs=dict(max_steps=n_steps, same_goals=same_goals, action_history=ac_hist)
          )
          v_num += 1

# THREE FINGER ENVS
  if three_finger:
    register(
      id="ThreeFingerRawEnv-v0",
      entry_point=three_finger.ThreeFingerRawEnv,
      max_episode_steps=100,
      kwargs={'max_steps': 100}
    )

    register(
      id="ThreeFingerRawResetEnv-v0",
      entry_point=three_finger.ThreeFingerRawEnv,
      max_episode_steps=100,
      kwargs={'max_steps': 100, 'reset_on_drop': True}
    )
