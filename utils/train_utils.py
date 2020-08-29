import gin
import time
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()  # ensures v2 enabled, >=1.15 compatibility

from absl import logging
from safemrl.utils import data_utils
from tf_agents.utils import common, nest_utils


def get_target_updater(sc_net, target_sc_net,
                       tau=0.005, period=1.,
                       name='update_target_sc_offline'):
  with tf.name_scope(name):
    def update():
      """Update target network."""
      critic_update = common.soft_variables_update(
        sc_net.variables,
        target_sc_net.variables, tau)
      return critic_update

    return common.Periodically(update, period, 'target_update')


def eval_safety(safety_critic, get_action, time_steps, use_sigmoid=True):
  obs = time_steps.observation
  ac = get_action(time_steps)
  sc_input = (obs, ac)
  q_val, _ = safety_critic(sc_input, time_steps.step_type, training=True)
  if use_sigmoid:
    q_safe = tf.nn.sigmoid(q_val)
  else:
    q_safe = q_val
  return q_safe


def get_train_step(agent, replay_buffer, batch_size=256):
  dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
  iterator = iter(dataset)

  @common.function
  def train_step():
    experience, _ = next(iterator)
    ret = agent.train(experience)
    return ret

  return train_step


def get_critic_train_step(agent, replay_buffer, sc_buffer, batch_size=256,
                          updating_sc=False, metrics=None):
  sc_dataset_pos = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size // 2,
    num_steps=2).prefetch(3)
  sc_dataset_neg = sc_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size // 2,
    num_steps=2).prefetch(3)
  sc_iter_pos = iter(sc_dataset_pos)
  sc_iter_neg = iter(sc_dataset_neg)
  dataset_spec = sc_dataset_neg.unbatch().element_spec[0]
  sc_buffer_last_id = common.function_in_tf1()(sc_buffer._get_last_id)

  @common.function(autograph=True)
  def critic_train_step():
    """Builds critic training step. Only evaluates if not updating_sc"""
    start_time = time.time()
    pos_experience, _ = next(sc_iter_pos)
    if sc_buffer_last_id() > batch_size // 2:
      neg_experience, _ = next(sc_iter_neg)
    else:
      neg_experience, _ = next(sc_iter_pos)
    experience = data_utils.concat_batches(pos_experience, neg_experience,
                                           dataset_spec)
    boundary_mask = tf.logical_not(experience.is_boundary()[:, 0])
    experience = nest_utils.fast_map_structure(
      lambda *x: tf.boolean_mask(*x, boundary_mask), experience)

    safe_rew = experience.observation['task_agn_rew'][:, 1]
    sc_weight = None
    if agent._fail_weight:
      sc_weight = tf.where(tf.cast(safe_rew, tf.bool),
                           agent._fail_weight / 0.5,
                           (1 - agent._fail_weight) / 0.5)
    ret = agent.train_sc(experience, safe_rew, weights=sc_weight,
                         metrics=metrics, training=updating_sc)
    logging.debug('critic train step: %4.2f sec', time.time() - start_time)
    return ret
  return critic_train_step


@gin.configurable
class MinitaurTerminationFn:
  def __init__(self, speed_metric, total_falls_metric, env_steps_metric,
               num_steps=10000, goal_speed=0.6, goal_speed_eps=0.05):
    self._speed_metric = speed_metric
    self._falls_metric = total_falls_metric
    self._steps_metric = env_steps_metric
    self._steps_since_fall = env_steps_metric.result().numpy()
    self._last_num_falls = total_falls_metric.result().numpy()
    self._num_steps = num_steps
    self._goal_speed = goal_speed
    self._goal_speed_eps = goal_speed_eps

  def __call__(self, *args, **kwargs):
    num_falls = self._falls_metric.result().numpy()
    if num_falls > self._last_num_falls:
      self._last_num_falls = num_falls
      self._steps_since_fall = self._steps_metric.result().numpy()
      return False
    if (self._goal_speed - self._goal_speed_eps <=
            self._speed_metric.result().numpy()
        and self._steps_metric.result().numpy() - self._steps_since_fall >
            self._num_steps):
      return True
    return False
