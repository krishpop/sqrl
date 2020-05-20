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

# Lint as: python3
"""Miscellanious utils for loading TF-Agent objects and env visualization.

Convenience methods to enable lightweight usage of TF-Agents library, env
visualization, and related uses.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import datetime
import os.path as osp
import matplotlib.pyplot as plt
import tensorflow as tf
import gin
import numpy as np
import wandb
import functools

from gym.wrappers import Monitor
from tf_agents.utils import common


# def construct_tf_agent(agent_class):
#   if agent_class

AGENT_CLASS_BINDINGS = {
  'sac-safe': 'safe_sac_agent.SafeSacAgent',
  'sac-safe-online': 'safe_sac_agent.SqrlAgent',
  'sac': 'sac_agent.SacAgent',
  'sac-ensemble': 'ensemble_sac_agent.EnsembleSacAgent'
}


def load_rb_ckpt(ckpt_dir, replay_buffer, ckpt_step=None):
  rb_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, max_to_keep=5, replay_buffer=replay_buffer)
  if ckpt_step is None:
    rb_checkpointer.initialize_or_restore().assert_existing_objects_matched()
  else:
    rb_checkpointer._checkpoint.restore(  # pylint: disable=protected-access
        osp.join(ckpt_dir, 'ckpt-{}'.format(ckpt_step)))
    rb_checkpointer._load_status.assert_existing_objects_matched()  # pylint: disable=protected-access
  return replay_buffer


def load_agent_ckpt(ckpt_dir, tf_agent, global_step=None):
  if global_step is None:
    global_step = tf.compat.v1.train.get_or_create_global_step()
  train_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, agent=tf_agent, global_step=global_step)
  train_checkpointer.initialize_or_restore()
  return tf_agent, global_step


def cleanup_checkpoints(checkpoint_dir):
  checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint_state is None:
    return
  for checkpoint_path in checkpoint_state.all_model_checkpoint_paths:
    tf.compat.v1.train.remove_checkpoint(checkpoint_path)
  return


def copy_rb(rb_s, rb_t):
  for x1, x2 in zip(rb_s.variables(), rb_t.variables()):
    x2.assign(x1)
  return rb_t


def load_pi_ckpt(ckpt_dir, policy):
  policy_checkpointer = common.Checkpointer(
      ckpt_dir=ckpt_dir, max_to_keep=1, policy=policy)
  policy_checkpointer.initialize_or_restore().assert_existing_objects_matched()
  return policy


def load_policies(policy, base_path, independent_runs):
  pi_loaded = []
  for run in independent_runs:
    pi_ckpt_path = osp.join(base_path, run, 'train/policy/')
    pi_loaded.append(load_pi_ckpt(pi_ckpt_path, policy))
  return pi_loaded


def create_default_writer_and_save_dir(root_dir):
  """Creates default directories."""
  base_dir = osp.expanduser(root_dir)
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  tag = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
  tb_logdir = osp.join(base_dir, tag, 'tb')
  save_dir = osp.join(base_dir, tag, 'train')
  tf.io.gfile.makedirs(tb_logdir)
  tf.io.gfile.makedirs(save_dir)
  writer = tf.contrib.summary.create_file_writer(tb_logdir)
  writer.set_as_default()
  return writer, save_dir


def record_point_mass_episode(tf_env, tf_policy, step=None, log_key='trajectory'):
  """Records summaries."""
  time_step = tf_env.reset()
  policy_state = tf_policy.get_initial_state(1)
  states, actions = [], []
  while not time_step.is_last():
    action_step = tf_policy.action(time_step, policy_state)
    a = action_step.action.numpy()[0]
    actions.append(a)
    s = time_step.observation['observation'].numpy()[0]
    states.append(s)
    policy_state = action_step.state
    time_step = tf_env.step(action_step.action)

  s = time_step.observation['observation'].numpy()[0]
  states.append(s)
  states = np.array(states)
  env = tf_env.pyenv.envs[0]

  f = plot_point_mass(env, states)
  wandb.log({log_key: wandb.Image(f)}, step=step)
  plt.close(f)
  return


def plot_point_mass(env, states):
  walls = env.walls.copy().astype(float)
  w, h = walls.shape
  start = env.unwrapped._start
  goal = env._goal
  walls[np.where(env.walls == 0)], walls[np.where(env.walls == 1)] = .5, 0.0

  f, ax = plt.subplots(figsize=(w * .8, h * .8))
  ax.scatter([start[1] - .5], [start[0] - .5], c='r', marker='X', s=150)
  ax.scatter([goal[1] - .5], [goal[0] - .5], c='g', s=150, alpha=.5)
  ax.matshow(walls, cmap=plt.cm.RdBu, vmin=0, vmax=1)
  u1, v1 = (states[1:, 1] - states[:-1, 1]) * .35, (states[:-1, 0] - states[1:, 0]) * .35
  ax.quiver(states[:-1, 1] - .5, states[:-1, 0] - .5, u1, v1, color='g',
            scale=.5, scale_units='x', minlength=min(np.abs(u1) + np.abs(v1)),
            headlength=3, headaxislength=2.5)
  ax.scatter(states[:, 1] - .5, states[:, 0] - .5)
  ax.set_axis_off()
  return f

def process_replay_buffer(replay_buffer, max_ep_len=500, k=1, as_tensor=True):
  """Process replay buffer to infer safety rewards with episode boundaries."""
  rb_data = replay_buffer.gather_all()
  rew = rb_data.reward

  boundary_idx = np.where(rb_data.is_boundary().numpy())[1]

  last_idx = 0
  k_labels = []

  for term_idx in boundary_idx:
    # TODO: remove +1?
    fail = 1 - int(term_idx - last_idx >= max_ep_len + 1)
    ep_rew = tf.gather(rew, np.arange(last_idx, term_idx), axis=1)
    labels = np.zeros(ep_rew.shape_as_list())  # ignore obs dim
    labels[:, -k:] = fail
    k_labels.append(labels)
    last_idx = term_idx

  flat_labels = np.concatenate(k_labels, axis=-1).astype(np.float32)
  n_flat_labels = flat_labels.shape[1]
  n_rews = rb_data.reward.shape_as_list()[1]
  safe_rew_labels = np.pad(
      flat_labels, ((0, 0), (0, n_rews - n_flat_labels)), mode='constant')
  if as_tensor:
    return tf.to_float(safe_rew_labels)
  return safe_rew_labels


# Pre-processor layers to remove observation from observation dict returned by
# goal-conditioned point-mass environment.
@gin.configurable
def extract_obs_merge_w_ac_layer():
  def f(layer_input):
    return tf.keras.layers.concatenate(
        [layer_input[0]['observation'], layer_input[1]], axis=1)
  return tf.keras.layers.Lambda(f)


# HACK: inputs to concatenate have to be in list (not tuple) format
# see "tensorflow_core/python/keras/layers/merge.py", line 378
@gin.configurable
def merge_obs_w_ac_layer():
  def f(layer_input):
    return tf.keras.layers.concatenate(list(layer_input), axis=-1)
  return tf.keras.layers.Lambda(f)


@gin.configurable
def extract_observation_layer():
  return tf.keras.layers.Lambda(lambda obs: obs['observation'])


@gin.configurable
def extract_observation_goal_layer():
  return tf.keras.layers.Lambda(lambda obs: tf.keras.layers.concatenate(
    [obs['observation'], obs['goal']]))


@gin.configurable
def monitor_freq(freq=100, vid_dir='./videos'):
  return functools.partial(Monitor, video_callable=lambda x: (x%freq) == 0,
                           directory=vid_dir, force=True)
