import wandb
import tensorflow as tf

def generate_tensor_summaries(tag, tensor, step, **kwargs):
  """Generates various summaries of `tensor` such as histogram, max, min, etc.

  Args:
    tag: A namescope tag for the summaries.
    tensor: The tensor to generate summaries of.
    step: Variable to use for summaries.
    **kwargs: Additional values to log
  """
  log_dict = {
      '{}/histogram'.format(tag): tensor,
      '{}/mean'.format(tag): tf.reduce_mean(tensor),
      '{}/mean_abs'.format(tag): tf.reduce_mean(tf.abs(tensor)),
      '{}/max'.format(tag): tf.reduce_max(tensor),
      '{}/min'.format(tag): tf.reduce_min(tensor)
  }
  for k, v in kwargs:
    log_dict['{}/{}'.format(tag, k)] = v
  wandb.log(log_dict, step=step)
