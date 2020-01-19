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
from safemrl.envs import minitaur
from safemrl.envs import cube_env
from safemrl.envs import env_randomizers
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_terrain_randomizer
from pybullet_envs.minitaur.envs.env_randomizers import minitaur_env_randomizer_from_config
from gym.envs.registration import register

registered = "MinitaurGoalVelocityEnv-v0" in gym.envs.registry.env_specs

if not registered:
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

  v_num = 0
  for ac_hist in [0, 7]:
    for n_steps in [500, 100, 1000]:
      for same_goals in [True, False]:
        register(
          id="SafemrlCube-v{}".format(v_num),
          entry_point=cube_env.SafemrlCubeEnv,
          max_episode_steps=n_steps,
          kwargs=dict(max_steps=n_steps,same_goals=same_goals, action_history=ac_hist)
        )
        v_num += 1
