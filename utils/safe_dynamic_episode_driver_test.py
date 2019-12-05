import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import episodic_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

from safemrl.utils import safe_dynamic_episode_driver
from safemrl.envs import point_mass

tf.enable_v2_behavior()

def test():
  num_episodes = 5
  py_env = parallel_py_environment.ParallelPyEnvironment([
    lambda: point_mass.env_load_fn() for _ in range(num_episodes)])
  env = tf_py_environment.TFPyEnvironment(py_env)
  policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(), env.action_spec())

  traj_spec = trajectory.from_transition(env.time_step_spec(), policy.policy_step_spec,
                                         env.time_step_spec())
  rb = episodic_replay_buffer.EpisodicReplayBuffer(traj_spec)
  srb = episodic_replay_buffer.StatefulEpisodicReplayBuffer(rb, num_episodes=num_episodes)
  rb2 = tf_uniform_replay_buffer.TFUniformReplayBuffer(traj_spec, 1)

  driver = safe_dynamic_episode_driver.SafeDynamicEpisodeDriver(env, policy, rb, rb2,
                                                                observers=[srb.add_batch],
                                                                num_episodes=num_episodes)
  driver.run()

if __name__ == "__main__":
  test()