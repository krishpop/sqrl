try:
  import highway_env.envs
  imported_highway = True
except ImportError:
  imported_highway = False
  print('could not import highway_env')

try:
  import pddm.envs
except ImportError:
  pass

import safemrl.envs
import cProfile
import pstats
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()
from gym.wrappers import FlattenObservation
from tf_agents.environments import suite_gym
from tf_agents.environments import wrappers
from tf_agents.policies import random_py_policy
from tf_agents.drivers import py_driver
import gin


def profile_env(env_str, max_ep_len, n_steps=None, env_wrappers=[]):
  n_steps = n_steps or max_ep_len * 2
  profile = [None]

  def profile_fn(p):
    assert isinstance(p, cProfile.Profile)
    profile[0] = p

  py_env = suite_gym.load(env_str, gym_env_wrappers=env_wrappers,
                          max_episode_steps=max_ep_len)
  env = wrappers.PerformanceProfiler(
    py_env, process_profile_fn=profile_fn,
    process_steps=n_steps)
  policy = random_py_policy.RandomPyPolicy(env.time_step_spec(), env.action_spec())

  driver = py_driver.PyDriver(env, policy, [], max_steps=n_steps)
  time_step = env.reset()
  policy_state = policy.get_initial_state()
  for _ in range(n_steps):
    time_step, policy_state = driver.run(time_step, policy_state)
  stats = pstats.Stats(profile[0])
  stats.print_stats()


# profile_env('SafemrlCube-v0', 500, env_wrappers=(cube_env.CubeTaskAgnWrapper,))
# profile_env('highway-v0', 40, env_wrappers=(FlattenObservation,
#                                             highway.ContAcWrapper,
#                                             highway.TaskAgnWrapper,))
# profile_env("MinitaurGoalVelocityEnv-v0", 500, env_wrappers=(minitaur.TaskAgnWrapper,))
# profile_env("SafeExpPointEnv-v0", 30)
