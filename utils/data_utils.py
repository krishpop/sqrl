import numpy as np
import time
import tensorflow as tf  # version >=2.0

from tf_agents.utils import nest_utils

# benchmarks dataset speed
def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in iter(dataset):
            # Performing a training step
            time.sleep(0.01)
    tf.print("Execution time:", time.perf_counter() - start_time)


def get_rand_batch(data, data_spec, mask=None, batch_size=128):
  """
  Gets random batch from Trajectory data extracted from replay_buffer.as_dataset

  :param data: tf_agents Trajectory instance, with nested elements of shape (n, )
  :param data_spec: tf_agent.collect_data_spec, for nested Trajectory elements
  :param batch_size: int, default is 128
  :return: batch from data
  """
  if mask is not None:
    data = nest_utils.fast_map_structure(lambda *x: tf.boolean_mask(*x, mask),
                                         data)
  n = nest_utils.get_outer_shape(data, data_spec).numpy()[0]
  mask = np.repeat(False, n)
  mask[np.random.choice(np.arange(n), batch_size, replace=False)] = True
  batch = nest_utils.fast_map_structure(lambda *x: tf.boolean_mask(*x, mask),
                                        data)
  return batch


def concat_batches(batch1, batch2, data_spec):
  batch1, batch2 = tf.nest.flatten(batch1), tf.nest.flatten(batch2)
  exp = tf.nest.pack_sequence_as(data_spec,
                                 [tf.concat([x, y], axis=0) for x, y in
                                  zip(batch1, batch2)])
  return exp


def copy_rb(rb_s, rb_t, filter_mask=None):
  """
  Copies replay buffer from rb_s to rb_t, one variable at a time
  :param rb_s: source replay buffer (copying from)
  :param rb_t: target replay buffer (copying to, usually empty)
  :param filter_mask: mask to put on source replay buffer to prevent all values
    from being copied
  :return: target replay buffer
  """
  max_len_s, max_len_t = rb_s._max_length, rb_t._max_length
  for x1, x2 in zip(rb_s.variables(), rb_t.variables()):
    varname = x1.name.split('/')[-1].rstrip(':0')
    if varname != 'last_id':
      if filter_mask is not None:
        assert filter_mask.dtype == tf.bool, 'filter_mask must be dtype tf.bool'
        x1 = tf.boolean_mask(x1, filter_mask)
    if varname == 'last_id' or max_len_t == x1.shape[0]:
      x2.assign(x1)
    else:
      x2[:max_len_s].assign(x1)
  return rb_t


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
