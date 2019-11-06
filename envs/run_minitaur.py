import gym
import gin
import os.path as osp
import tensorflow as tf
import imageio
import numpy as np

from tf_agents.agents.sac import sac_agent
from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.utils import common

import envs
import algos
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from absl import logging

tf.compat.v1.enable_v2_behavior()

flags.DEFINE_string('load_dir', None, 'load path for policy')
flags.DEFINE_string('vid_dir', '../videos/sac/', 'save path for video')
flags.DEFINE_integer('n_episodes', -1, 'number of episodes to simulate')
flags.DEFINE_bool('save_vid', False, 'whether or not to save episode rollout')
flags.DEFINE_bool('debug', False, 'turn on debugging for simulator')
flags.DEFINE_bool('render', False, 'turn on rendering for simulator')
flags.DEFINE_bool('random_policy', False, 'whether or not to run random policy')
flags.DEFINE_string('urdf', 'rainbow_dash_v0', 'urdf file to  use when loading env')
flags.DEFINE_integer('history_length', 6, 'number of past obs to include in state')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding to pass through.')


FLAGS = flags.FLAGS


@gin.configurable
def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
  del init_action_stddev
  return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True)


def load_policy(tf_env):
  load_dir = FLAGS.load_dir
  assert load_dir and osp.exists(load_dir), 'need to provide valid load_dir to load policy, got: {}'.format(load_dir)
  global_step = tf.compat.v1.train.get_or_create_global_step()
  time_step_spec = tf_env.time_step_spec()
  observation_spec = time_step_spec.observation
  action_spec = tf_env.action_spec()

  actor_net = actor_distribution_network.ActorDistributionNetwork(
          observation_spec,
          action_spec,
          fc_layer_params=(256, 256),
          continuous_projection_net=normal_projection_net)

  critic_net = critic_network.CriticNetwork(
      (observation_spec, action_spec),
      joint_fc_layer_params=(256, 256))

  tf_agent = sac_agent.SacAgent(
      time_step_spec,
      action_spec,
      actor_network=actor_net,
      critic_network=critic_net,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=3e-4),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=3e-4),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
          learning_rate=3e-4),
      target_update_tau=0.005,
      target_update_period=1,
      td_errors_loss_fn=tf.keras.losses.mse,
      gamma=0,
      reward_scale_factor=1.,
      gradient_clipping=1.,
      debug_summaries=False,
      summarize_grads_and_vars=False,
      train_step_counter=global_step)

  train_checkpointer = common.Checkpointer(
    ckpt_dir=load_dir,
    agent=tf_agent,
    global_step=global_step
  )
  status = train_checkpointer.initialize_or_restore()
  status.expect_partial()
  logging.info('Loaded from checkpoint: %s, trained %s steps',
               train_checkpointer._manager.latest_checkpoint,
               global_step.numpy())
  return tf_agent.policy

def save_vid(frames, vid_path):
  writer = imageio.get_writer(vid_path)
  for frame in frames:
    writer.append_data(frame)
  writer.close()

def run_loaded_policy(env, tf_env, policy):
  traj_len = 0
  max_speed = 0.
  mean_vel = tf.keras.metrics.Mean(name='mean_vel')
  time_step = tf_env.reset()
  if FLAGS.save_vid:
    img = env.render('rgb_array')
    ax = plt.imshow(img)
    plt.text(450, 15, '{:3.2f}'.format(env.unwrapped._current_vel))
    frames = [img]
  elif FLAGS.render:
    env.render()
  pol_state = policy.get_initial_state(1)
  while not time_step.is_last():
    action_step = policy.action(time_step, pol_state)
    action, pol_state = action_step.action, action_step.state
    time_step = tf_env.step(action)
    if FLAGS.save_vid:
      frames.append(env.render('rgb_array'))
    elif FLAGS.render:
      env.render()
    if traj_len > 100:
      mean_vel.update_state(env.unwrapped._current_vel)
    if abs(max_speed) < abs(env.unwrapped._current_vel):
      max_speed = env.unwrapped._current_vel
    traj_len += 1
  if FLAGS.debug:
    logging.info('ran %s steps', traj_len)
  if FLAGS.save_vid:
    i = 0
    vid_path = lambda: osp.join(FLAGS.vid_dir,
                                'episode-{}.mp4'.format(i))
    while osp.exists(vid_path()):
      i += 1
    vid_path = vid_path()
    save_vid(frames, vid_path)
  mean_vel_ = mean_vel.result().numpy()
  mean_vel.reset_states()
  return max_speed, mean_vel_, traj_len

def run_random_policy(env):
  env.reset()
  d = False
  max_speed = 0.
  mean_vel = tf.keras.metrics.Mean(name='mean_vel')
  traj_len = 0
  while not d:
    env.render()
    o, r, d, i = env.step(env.action_space.sample())
    if abs(max_speed) < abs(env.unwrapped._current_vel):
      max_speed = env.unwrapped._current_vel
    mean_vel.update_state(env.unwrapped._current_vel)
    traj_len += 1
  mean_vel_ = mean_vel.result().numpy()
  mean_vel.reset_states()
  return max_speed, mean_vel_, traj_len

def run_episodes():
  env = gym.make('MinitaurGoalVelocityEnv-v0', render=FLAGS.render,
                 debug=FLAGS.debug)
  max_speeds = []
  mean_speeds = []
  traj_lens = []
  try:
    i = 0
    if FLAGS.random_policy:
      run_episode = lambda: run_random_policy(env)
    else:
      tf_env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(env))
      policy = load_policy(tf_env)
      run_episode = lambda: run_loaded_policy(env, tf_env, policy)
    while i != FLAGS.n_episodes:
      ret = run_episode()
      i += 1
      max_speed, mean_speed, traj_len = ret
      max_speeds.append(max_speed)
      mean_speeds.append(mean_speed)
      traj_lens.append(traj_len)
  except KeyboardInterrupt:
    logging.info('Exiting')
    env.close()

  logging.info('max speeds (mean, min, max, std): %s, %s, %s, %s', np.mean(max_speeds), np.min(max_speeds),
        np.max(max_speeds), np.std(max_speeds))
  logging.info('mean speeds (mean, min, max, std): %s, %s, %s, %s', np.mean(mean_speeds), np.min(mean_speeds),
               np.max(mean_speeds), np.std(mean_speeds))
  logging.info('traj lens (mean, min, max, std): %s, %s, %s, %s', np.mean(traj_lens), np.min(traj_lens),
        np.max(traj_lens), np.std(traj_lens))


def main(_):
  logging.info('parsing config files: %s', FLAGS.gin_file)
  gin.parse_config_files_and_bindings(
    FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
  run_episodes()


if __name__ == "__main__":
  app.run(main)