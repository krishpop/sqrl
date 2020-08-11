import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.trajectories import trajectory


def plot_fail_prob(init_step, after_init_step,
                   before_fail_step, fail_step,
                   tf_agent=None, pol_eval=False,
                   safety_critic=None):
  assert pol_eval is False or tf_agent is not None, 'if pol_eval is True, need to include tf_agent'
  assert safety_critic is not None or tf_agent is not None, 'if safety_critic is None, need to include tf_agent'
  safety_critic = safety_critic or tf_agent._safety_critic_network
  if pol_eval:
    time_step, _, _ = trajectory.to_transition(before_fail_step, fail_step)
    ac = tf_agent.policy.action(time_step, ()).action
  else:
    ac = fail_step.action
  sc_input = (fail_step.observation, ac)
  p_fail_logits, _ = safety_critic(sc_input, fail_step.step_type, training=True)
  p_fail = tf.nn.sigmoid(p_fail_logits)

  if pol_eval:
    time_step, _, _ = trajectory.to_transition(init_step, after_init_step)
    ac = tf_agent.policy.action(time_step, ()).action
  else:
    ac = init_step.action

  sc_input = (init_step.observation, init_step.action)
  first_p_fail_logits, _ = safety_critic(sc_input, init_step.step_type,
                                         training=True)
  first_p_fail = tf.nn.sigmoid(first_p_fail_logits)

  f, ax = plt.subplots(1, 2, figsize=(15, 6))
  ax[0].hist(p_fail, bins=20)
  ax[0].set_title('Failure state failure prob')
  ax[1].hist(first_p_fail, bins=20)
  ax[1].set_title('Initial state failure prob')
  plt.show()
  return p_fail, first_p_fail


def embed_mp4(filename):
  import base64
  import IPython
  """Embeds an mp4 file in the notebook."""
  video = open(filename, 'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
  <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)